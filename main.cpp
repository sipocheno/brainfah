#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <sys/wait.h>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <print>
#include <set>
#include <csetjmp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

constexpr std::size_t TAPE_SIZE{16ULL << 20}; // 16 MB total tape (8 MB negative + 8 MB positive)
constexpr std::size_t TAPE_OFFSET{8ULL << 20}; // Start pointer right in the middle of the tape
constexpr std::size_t GUARD_SIZE{64ULL << 20}; // 64 MB PROT_NONE hardware guard zones
constexpr std::size_t MAX_PRE_STEPS{500'000}; // Fast compile-time partial evaluation budget

namespace {
    enum class Op : std::uint8_t {
        Add, // p[off] += arg
        Set, // p[off] = arg
        Move, // p += arg
        Out, // out(p[off])
        In, // p[off] = in()
        JumpIfZero, // while (p[off]) {
        JumpIfNotZero, // }
        IfNonZero, // if (p[off]) {
        EndIf, // }
        MulAdd, // p[off2] += arg * p[off]
        MulAddClear, // p[off2] += arg * p[off]; p[off] = 0
        MulCells, // p[off2] += arg * p[off] * p[off3]
        Scan, // while (*p) p += arg
        MoveJnz, // p += off2; }
        Halt
    };

    struct Ir {
        Op op;
        int arg{};
        int off{};
        int off2{};
        int off3{};
    };

    struct Loop {
        std::size_t open;
        int base;
    };

    struct ProgramIr {
        std::vector<Ir> code;
        std::string pre_output;
    };

    constexpr int mod256(const int x) noexcept {
        return x & 0xFF;
    }

    constexpr int inverse256(const int a) noexcept {
        const auto ua = static_cast<unsigned>(a);
        unsigned x = ua;
        x *= 2U - ua * x;
        x *= 2U - ua * x;
        x *= 2U - ua * x;
        return static_cast<int>(x & 0xFFU);
    }

    struct CellEffect {
        bool is_set{false};
        int val{0};
    };

    // Materializes `Move(base)` right before `open` and shifts all offsets inside `[open .. code.size())` by `-base`.
    // Must ONLY be called when no `Move`/`Scan`/`MoveJnz` exist inside `[open .. code.size())` yet!
    void rebase_loop_to_zero(std::vector<Ir> &code, Loop &loop) {
        if (loop.base == 0) return;
        const int b = loop.base;
        for (std::size_t k = loop.open; k < code.size(); ++k) {
            auto &[op, arg, off, off2, off3] = code[k];
            if (op == Op::MulCells) {
                off -= b;
                off2 -= b;
                off3 -= b;
            } else if (op == Op::MulAdd || op == Op::MulAddClear) {
                off -= b;
                off2 -= b;
            } else if (op != Op::Move && op != Op::Scan && op != Op::Halt) {
                off -= b;
            }
            if (op == Op::JumpIfZero || op == Op::JumpIfNotZero ||
                op == Op::IfNonZero || op == Op::EndIf || op == Op::MoveJnz) {
                arg += 1;
            }
        }
        if (!code.empty() && loop.open > 0 && code[loop.open - 1].op == Op::Move) {
            code[loop.open - 1].arg += b;
            // Shift back jump targets since we merged into existing Move instead of inserting
            for (std::size_t k = loop.open; k < code.size(); ++k) {
                if (Ir &ins = code[k]; ins.op == Op::JumpIfZero || ins.op == Op::JumpIfNotZero ||
                                       ins.op == Op::IfNonZero || ins.op == Op::EndIf || ins.op == Op::MoveJnz) {
                    ins.arg -= 1;
                }
            }
            if (code[loop.open - 1].arg == 0) {
                code.erase(code.begin() + static_cast<std::ptrdiff_t>(loop.open - 1));
                --loop.open;
                for (std::size_t k = loop.open; k < code.size(); ++k) {
                    if (Ir &ins = code[k]; ins.op == Op::JumpIfZero || ins.op == Op::JumpIfNotZero ||
                                           ins.op == Op::IfNonZero || ins.op == Op::EndIf || ins.op == Op::MoveJnz) {
                        ins.arg -= 1;
                    }
                }
            }
        } else {
            code.insert(code.begin() + static_cast<std::ptrdiff_t>(loop.open), Ir{.op = Op::Move, .arg = b});
            ++loop.open;
        }
        loop.base = 0;
    }

    // Affine summary of a straight-line loop body: cell -> (sum of coef*entry_cell + constant), all mod 256.
    struct Affine {
        std::map<int, int> coef;
        int c{0};
    };

    void affine_add_scaled(Affine &dst, const Affine src, const int k) {
        for (const auto &[cell, a]: src.coef) {
            const int v = mod256(dst.coef[cell] + k * a);
            if (v == 0) dst.coef.erase(cell);
            else dst.coef[cell] = v;
        }
        dst.c = mod256(dst.c + k * src.c);
    }

    // Closed form for counting loops whose body is straight-line Add/Set/MulAdd/MulAddClear, e.g. the classic
    // multiplication idiom x[->y[->+>+<<]>>[-<<+>>]<<<]. The first iteration is kept verbatim (it may see arbitrary
    // values in scratch cells); all later iterations are summarized: scratch cells become constants, the counter
    // steps by an odd constant d, and every other changed cell is an accumulator of invariant cells, so
    //   n = -counter * inv(d)  iterations  =>  acc += n * (b + sum a_z * z)
    // which needs only MulAdd and the cell-product op MulCells.
    bool try_optimize_affine(std::vector<Ir> &code, const std::size_t open, const int base) {
        if (code.size() - open > 48) return false;

        std::map<int, Affine> f;
        const auto get = [&f](const int cell) -> Affine & {
            auto [it, inserted] = f.try_emplace(cell);
            if (inserted) it->second.coef[cell] = 1;
            return it->second;
        };

        for (std::size_t i = open + 1; i < code.size(); ++i) {
            const Ir &ins = code[i];
            switch (ins.op) {
                case Op::Add: {
                    Affine &e = get(ins.off);
                    e.c = mod256(e.c + ins.arg);
                    break;
                }
                case Op::Set:
                    f[ins.off] = Affine{.coef = {}, .c = mod256(ins.arg)};
                    break;
                case Op::MulAdd:
                case Op::MulAddClear: {
                    if (ins.off == ins.off2) return false;
                    const Affine src = get(ins.off);
                    affine_add_scaled(get(ins.off2), src, ins.arg);
                    if (ins.op == Op::MulAddClear) f[ins.off] = Affine{.coef = {}, .c = 0};
                    break;
                }
                default:
                    return false;
            }
        }

        if (!f.contains(base)) return false;

        // Cells that end every iteration with a constant (cleared / Set): their entry value is that constant
        // from the second iteration on.
        std::map<int, int> known;
        for (const auto &[cell, e]: f) {
            if (e.coef.empty()) known[cell] = e.c;
        }
        if (known.contains(base)) return false;

        std::map<int, Affine> g;
        for (const auto &[cell, e]: f) {
            if (known.contains(cell)) continue;
            Affine ne;
            ne.c = e.c;
            for (const auto &[z, a]: e.coef) {
                if (const auto kz = known.find(z); kz != known.end()) {
                    ne.c = mod256(ne.c + a * kz->second);
                } else {
                    ne.coef[z] = a;
                }
            }
            g[cell] = std::move(ne);
        }

        const Affine &gb = g.at(base);
        if (gb.coef.size() != 1 || gb.coef.begin()->first != base || gb.coef.begin()->second != 1 ||
            (gb.c & 1) == 0) {
            return false;
        }
        const int d = gb.c;

        std::set<int> acc;
        for (const auto &[cell, e]: g) {
            if (cell == base) continue;
            const auto self = e.coef.find(cell);
            if (self == e.coef.end() || self->second != 1) return false; // copy-like or geometric: give up
            if (e.coef.size() == 1 && e.c == 0) continue; // invariant
            acc.insert(cell);
        }
        for (const int y: acc) {
            for (const auto &[z, a]: g.at(y).coef) {
                if (z == y) continue;
                if (z == base || acc.contains(z)) return false;
            }
        }

        const int nf = mod256(-inverse256(d));
        std::vector<Ir> tail;
        for (const int y: acc) {
            const Affine &e = g.at(y);
            if (const int k = mod256(nf * e.c); k != 0) {
                tail.push_back({.op = Op::MulAdd, .arg = k, .off = base, .off2 = y});
            }
            for (const auto &[z, a]: e.coef) {
                if (z == y) continue;
                if (const int k = mod256(nf * a); k != 0) {
                    tail.push_back({.op = Op::MulCells, .arg = k, .off = base, .off2 = y, .off3 = z});
                }
            }
        }
        tail.push_back({.op = Op::Set, .arg = 0, .off = base});

        code[open].op = Op::IfNonZero;
        code.insert(code.end(), tail.begin(), tail.end());
        code.push_back({.op = Op::EndIf, .arg = static_cast<int>(open), .off = base});
        code[open].arg = static_cast<int>(code.size() - 1);
        return true;
    }

    bool try_optimize_balanced(std::vector<Ir> &code, const std::size_t open, const int base) {
        std::map<int, CellEffect> effects;
        bool pure_linear = true;

        for (std::size_t i = open + 1; i < code.size(); ++i) {
            if (const Ir &ins = code[i]; ins.op == Op::Add) {
                auto &[is_set, val] = effects[ins.off];
                val = mod256(val + ins.arg);
            } else if (ins.op == Op::Set) {
                auto &[is_set, val] = effects[ins.off];
                is_set = true;
                val = mod256(ins.arg);
            } else {
                pure_linear = false;
                break;
            }
        }

        if (pure_linear) {
            const auto it = effects.find(base);
            if (it == effects.end()) return false;

            if (effects.size() == 1) {
                if ((it->second.is_set && it->second.val == 0) || (!it->second.is_set && (it->second.val & 1) != 0)) {
                    code.erase(code.begin() + static_cast<std::ptrdiff_t>(open), code.end());
                    code.push_back({.op = Op::Set, .arg = 0, .off = base});
                    return true;
                }
            }

            if (it->second.is_set && it->second.val == 0) {
                code[open].op = Op::IfNonZero;
                code.push_back({.op = Op::EndIf, .arg = static_cast<int>(open), .off = base});
                code[open].arg = static_cast<int>(code.size() - 1);
                return true;
            }

            if (!it->second.is_set && (it->second.val & 1) != 0) {
                bool can_mul = true;
                for (const auto &[k, eff]: effects) {
                    if (k != base && eff.is_set) {
                        can_mul = false;
                        break;
                    }
                }
                if (can_mul) {
                    const int inv = inverse256(it->second.val);
                    code.erase(code.begin() + static_cast<std::ptrdiff_t>(open), code.end());
                    for (const auto &[k, eff]: effects) {
                        if (k == base) continue;
                        const int factor = mod256(-eff.val * inv);
                        if (factor == 0) continue;
                        code.push_back({.op = Op::MulAdd, .arg = factor, .off = base, .off2 = k});
                    }
                    if (code.size() > open) {
                        code.back().op = Op::MulAddClear;
                    } else {
                        code.push_back({.op = Op::Set, .arg = 0, .off = base});
                    }
                    return true;
                }
            }
        }

        // Check if balanced loop unconditionally zeroes `base` at top-level of its body
        bool base_zeroed = false;
        int inner_depth = 0;
        for (std::size_t i = open + 1; i < code.size(); ++i) {
            const Ir &ins = code[i];
            if (ins.op == Op::Move || ins.op == Op::MoveJnz || ins.op == Op::Scan || ins.op == Op::In) {
                base_zeroed = false;
                break;
            }
            if (ins.op == Op::JumpIfZero || ins.op == Op::IfNonZero) {
                ++inner_depth;
                continue;
            }
            if (ins.op == Op::JumpIfNotZero || ins.op == Op::EndIf) {
                --inner_depth;
                // A loop/if on the base cell can only be left with that cell equal to zero.
                if (inner_depth == 0 && ins.off == base) base_zeroed = true;
                continue;
            }
            if (inner_depth == 0) {
                if ((ins.op == Op::Set && ins.off == base && mod256(ins.arg) == 0) ||
                    (ins.op == Op::MulAddClear && ins.off == base && ins.off2 != base)) {
                    base_zeroed = true;
                } else if (ins.off == base ||
                           ((ins.op == Op::MulAdd || ins.op == Op::MulAddClear || ins.op == Op::MulCells) &&
                            ins.off2 == base)) {
                    base_zeroed = false;
                }
            } else {
                if (ins.off == base || ((ins.op == Op::MulAdd || ins.op == Op::MulAddClear || ins.op == Op::MulCells) &&
                                        ins.off2 == base)) {
                    base_zeroed = false;
                }
            }
        }
        if (base_zeroed && inner_depth == 0) {
            code[open].op = Op::IfNonZero;
            code.push_back({.op = Op::EndIf, .arg = static_cast<int>(open), .off = base});
            code[open].arg = static_cast<int>(code.size() - 1);
            return true;
        }

        return try_optimize_affine(code, open, base);
    }

    int get_known_value(const std::vector<Ir> &code, const int off, const bool at_top_level,
                        const bool has_moved) noexcept {
        for (std::size_t i = code.size(); i-- > 0;) {
            const Ir &ins = code[i];
            switch (ins.op) {
                case Op::Set:
                    if (ins.off == off) return mod256(ins.arg);
                    break;
                case Op::MulAddClear:
                    if (ins.off == off && ins.off2 != off) return 0;
                    if (ins.off2 == off) return -1;
                    break;
                case Op::JumpIfNotZero:
                case Op::EndIf:
                    if (ins.off == off) return 0;
                    return -1;
                case Op::Add:
                    if (ins.off == off) return -1;
                    break;
                case Op::MulAdd:
                case Op::MulCells:
                    if (ins.off2 == off) return -1;
                    break;
                case Op::Out:
                    break;
                case Op::JumpIfZero:
                case Op::IfNonZero:
                case Op::Move:
                case Op::MoveJnz:
                case Op::Scan:
                case Op::In:
                case Op::Halt:
                    return -1;
            }
        }
        return (at_top_level && !has_moved) ? 0 : -1;
    }

    ProgramIr pre_evaluate(std::vector<Ir> &ir) {
        constexpr std::size_t LOCAL_TAPE_SIZE{256 * 1024};
        constexpr auto origin = static_cast<std::ptrdiff_t>(LOCAL_TAPE_SIZE / 2);
        std::vector<std::uint8_t> tape(LOCAL_TAPE_SIZE, 0);
        std::vector<std::uint8_t> saved_tape(LOCAL_TAPE_SIZE, 0);

        std::ptrdiff_t ptr = origin;
        std::string output;
        std::size_t steps = 0;
        std::size_t committed_pc = 0;
        std::ptrdiff_t min_touched = origin;
        std::ptrdiff_t max_touched = origin;

        const auto in_bounds = [](const std::ptrdiff_t idx) noexcept -> bool {
            return idx >= 0 && static_cast<std::size_t>(idx) < LOCAL_TAPE_SIZE;
        };

        const auto next_top_level = [&](const std::size_t pc_idx) noexcept -> std::size_t {
            if (ir[pc_idx].op == Op::JumpIfZero || ir[pc_idx].op == Op::IfNonZero) {
                return static_cast<std::size_t>(ir[pc_idx].arg) + 1;
            }
            return pc_idx + 1;
        };

        std::size_t pc = 0;

        while (pc < ir.size() && ir[pc].op != Op::Halt) {
            const std::size_t stmt_end = next_top_level(pc);
            const std::ptrdiff_t saved_ptr = ptr;
            const std::size_t saved_out_len = output.size();
            const std::ptrdiff_t saved_min = min_touched;
            const std::ptrdiff_t saved_max = max_touched;

            // Fast snapshot of the touched range only
            const auto copy_len = static_cast<std::size_t>(saved_max - saved_min + 1);
            std::memcpy(saved_tape.data() + saved_min, tape.data() + saved_min, copy_len);

            bool ok = true;
            std::size_t cur = pc;

            while (cur < stmt_end) {
                if (++steps > MAX_PRE_STEPS) {
                    ok = false;
                    break;
                }
                const auto &[op, arg, off, off2, off3] = ir[cur];
                switch (op) {
                    case Op::Add: {
                        const std::ptrdiff_t idx = ptr + off;
                        if (!in_bounds(idx)) {
                            ok = false;
                            break;
                        }
                        tape[static_cast<std::size_t>(idx)] = static_cast<std::uint8_t>(
                            tape[static_cast<std::size_t>(idx)] + static_cast<unsigned int>(arg));
                        min_touched = std::min(min_touched, idx);
                        max_touched = std::max(max_touched, idx);
                        ++cur;
                        break;
                    }
                    case Op::Set: {
                        const std::ptrdiff_t idx = ptr + off;
                        if (!in_bounds(idx)) {
                            ok = false;
                            break;
                        }
                        tape[static_cast<std::size_t>(idx)] = static_cast<std::uint8_t>(arg);
                        min_touched = std::min(min_touched, idx);
                        max_touched = std::max(max_touched, idx);
                        ++cur;
                        break;
                    }
                    case Op::Move:
                        ptr += arg;
                        if (!in_bounds(ptr)) {
                            ok = false;
                            break;
                        }
                        min_touched = std::min(min_touched, ptr);
                        max_touched = std::max(max_touched, ptr);
                        ++cur;
                        break;
                    case Op::Out: {
                        const std::ptrdiff_t idx = ptr + off;
                        if (!in_bounds(idx)) {
                            ok = false;
                            break;
                        }
                        output.push_back(static_cast<char>(tape[static_cast<std::size_t>(idx)]));
                        ++cur;
                        break;
                    }
                    case Op::In:
                        ok = false;
                        break;
                    case Op::MulAdd: {
                        const std::ptrdiff_t src = ptr + off;
                        const std::ptrdiff_t dst = ptr + off2;
                        if (!in_bounds(src) || !in_bounds(dst)) {
                            ok = false;
                            break;
                        }
                        const std::uint8_t sval = tape[static_cast<std::size_t>(src)];
                        if (sval != 0) {
                            tape[static_cast<std::size_t>(dst)] = static_cast<std::uint8_t>(
                                tape[static_cast<std::size_t>(dst)] + static_cast<unsigned int>(arg) * sval);
                            min_touched = std::min({min_touched, src, dst});
                            max_touched = std::max({max_touched, src, dst});
                        }
                        ++cur;
                        break;
                    }
                    case Op::MulAddClear: {
                        const std::ptrdiff_t src = ptr + off;
                        const std::ptrdiff_t dst = ptr + off2;
                        if (!in_bounds(src) || !in_bounds(dst)) {
                            ok = false;
                            break;
                        }
                        if (const std::uint8_t sval = tape[static_cast<std::size_t>(src)]; sval != 0) {
                            tape[static_cast<std::size_t>(dst)] = static_cast<std::uint8_t>(
                                tape[static_cast<std::size_t>(dst)] + static_cast<unsigned int>(arg) * sval);
                            tape[static_cast<std::size_t>(src)] = 0;
                            min_touched = std::min({min_touched, src, dst});
                            max_touched = std::max({max_touched, src, dst});
                        }
                        ++cur;
                        break;
                    }
                    case Op::MulCells: {
                        const std::ptrdiff_t a = ptr + off;
                        const std::ptrdiff_t b = ptr + off3;
                        const std::ptrdiff_t dst = ptr + off2;
                        if (!in_bounds(a) || !in_bounds(b) || !in_bounds(dst)) {
                            ok = false;
                            break;
                        }
                        const std::uint8_t av = tape[static_cast<std::size_t>(a)];
                        const std::uint8_t bv = tape[static_cast<std::size_t>(b)];
                        tape[static_cast<std::size_t>(dst)] = static_cast<std::uint8_t>(
                            tape[static_cast<std::size_t>(dst)] + static_cast<unsigned int>(arg) * av * bv);
                        min_touched = std::min({min_touched, a, b, dst});
                        max_touched = std::max({max_touched, a, b, dst});
                        ++cur;
                        break;
                    }
                    case Op::JumpIfZero:
                    case Op::IfNonZero: {
                        const std::ptrdiff_t idx = ptr + off;
                        if (!in_bounds(idx)) {
                            ok = false;
                            break;
                        }
                        if (tape[static_cast<std::size_t>(idx)] == 0) {
                            cur = static_cast<std::size_t>(arg) + 1;
                        } else {
                            ++cur;
                        }
                        break;
                    }
                    case Op::JumpIfNotZero: {
                        const std::ptrdiff_t idx = ptr + off;
                        if (!in_bounds(idx)) {
                            ok = false;
                            break;
                        }
                        if (tape[static_cast<std::size_t>(idx)] != 0) {
                            cur = static_cast<std::size_t>(arg) + 1;
                        } else {
                            ++cur;
                        }
                        break;
                    }
                    case Op::EndIf:
                        ++cur;
                        break;
                    case Op::MoveJnz: {
                        ptr += off2;
                        if (!in_bounds(ptr)) {
                            ok = false;
                            break;
                        }
                        min_touched = std::min(min_touched, ptr);
                        max_touched = std::max(max_touched, ptr);
                        if (tape[static_cast<std::size_t>(ptr)] != 0) {
                            cur = static_cast<std::size_t>(arg) + 1;
                        } else {
                            ++cur;
                        }
                        break;
                    }
                    case Op::Scan: {
                        if (!in_bounds(ptr)) {
                            ok = false;
                            break;
                        }
                        while (tape[static_cast<std::size_t>(ptr)] != 0) {
                            ptr += arg;
                            if (!in_bounds(ptr) || ++steps > MAX_PRE_STEPS) {
                                ok = false;
                                break;
                            }
                        }
                        if (ok) {
                            min_touched = std::min(min_touched, ptr);
                            max_touched = std::max(max_touched, ptr);
                            ++cur;
                        }
                        break;
                    }
                    case Op::Halt:
                        ok = false;
                        break;
                }
                if (!ok) break;
            }

            if (!ok) {
                // Zero out newly touched regions and restore snapshot
                std::memset(tape.data() + min_touched, 0, static_cast<std::size_t>(max_touched - min_touched + 1));
                std::memcpy(tape.data() + saved_min, saved_tape.data() + saved_min, copy_len);
                ptr = saved_ptr;
                output.resize(saved_out_len);
                min_touched = saved_min;
                max_touched = saved_max;
                break;
            }

            committed_pc = stmt_end;
            pc = stmt_end;
        }

        if (committed_pc == 0) {
            return ProgramIr{.code = std::move(ir), .pre_output = {}};
        }

        std::vector<Ir> new_ir;
        new_ir.reserve((ir.size() - committed_pc) + static_cast<std::size_t>(max_touched - min_touched) + 8);

        if (ir[committed_pc].op == Op::Halt) {
            new_ir.push_back({.op = Op::Halt});
            return ProgramIr{.code = std::move(new_ir), .pre_output = std::move(output)};
        }

        for (std::ptrdiff_t idx = min_touched; idx <= max_touched; ++idx) {
            const std::uint8_t v = tape[static_cast<std::size_t>(idx)];
            if (v != 0) {
                new_ir.push_back({.op = Op::Set, .arg = static_cast<int>(v), .off = static_cast<int>(idx - origin)});
            }
        }
        if (ptr != origin) {
            new_ir.push_back({.op = Op::Move, .arg = static_cast<int>(ptr - origin)});
        }

        const int shift_pc = static_cast<int>(new_ir.size()) - static_cast<int>(committed_pc);
        for (std::size_t i = committed_pc; i < ir.size(); ++i) {
            Ir ins = ir[i];
            if (ins.op == Op::JumpIfZero || ins.op == Op::JumpIfNotZero ||
                ins.op == Op::IfNonZero || ins.op == Op::EndIf || ins.op == Op::MoveJnz) {
                ins.arg += shift_pc;
            }
            new_ir.push_back(ins);
        }

        return ProgramIr{.code = std::move(new_ir), .pre_output = std::move(output)};
    }
}

static ProgramIr convert(const std::string_view program) {
    std::vector<Ir> result;
    std::vector<Loop> stack;
    result.reserve(program.size() / 2 + 32);

    int pending = 0;
    bool has_moved = false;

    const auto emit_move = [&](const int delta) {
        if (delta == 0) return;
        has_moved = true;
        if (!result.empty() && result.back().op == Op::Move) {
            result.back().arg += delta;
            if (result.back().arg == 0) result.pop_back();
        } else {
            result.push_back({.op = Op::Move, .arg = delta});
        }
    };

    // Before emitting ANY hardware pointer movement (Move, Scan, MoveJnz) inside enclosing loops,
    // we must ensure all currently open enclosing loops have their `base` rebased to 0 so that
    // their opening `while (p[0])` and closing `}` operate on the hardware pointer `p` directly!
    const auto add = [&](const int delta) {
        for (std::size_t i = result.size(); i-- > 0;) {
            auto &[op, arg, off, off2, off3] = result[i];
            if (op == Op::Add || op == Op::Set) {
                if (off == pending) {
                    arg = mod256(arg + delta);
                    if (op == Op::Add && arg == 0) {
                        result.erase(result.begin() + static_cast<std::ptrdiff_t>(i));
                    }
                    return;
                }
            } else if (op == Op::MulAdd || op == Op::MulAddClear) {
                if (off == pending || off2 == pending) break;
            } else {
                break;
            }
        }
        const int kv = get_known_value(result, pending, stack.empty(), has_moved);
        if (kv != -1) {
            result.push_back({.op = Op::Set, .arg = mod256(kv + delta), .off = pending});
        } else {
            result.push_back({.op = Op::Add, .arg = mod256(delta), .off = pending});
        }
    };

    const std::size_t length = program.size();
    for (std::size_t i = 0; i < length; ++i) {
        if (const char c = program[i]; c == '+') {
            add(1);
        } else if (c == '-') {
            add(-1);
        } else if (c == '>' || c == '<') {
            pending += (c == '>') ? 1 : -1;
        } else if (c == '.') {
            result.push_back({.op = Op::Out, .arg = 0, .off = pending});
        } else if (c == ',') {
            result.push_back({.op = Op::In, .arg = 0, .off = pending});
        } else if (c == '[') {
            if (get_known_value(result, pending, stack.empty(), has_moved) == 0) {
                std::size_t depth = 1;
                std::size_t j = i + 1;
                for (; j < length && depth > 0; ++j) {
                    if (program[j] == '[') ++depth;
                    else if (program[j] == ']') --depth;
                }
                if (depth > 0) throw std::runtime_error{"Unclosed ["};
                i = j - 1;
                continue;
            }
            stack.push_back({.open = result.size(), .base = pending});
            result.push_back({.op = Op::JumpIfZero, .arg = 0, .off = pending});
        } else if (c == ']') {
            if (stack.empty()) throw std::runtime_error{"Extra ]"};
            Loop loop = stack.back();
            stack.pop_back();

            bool body_modified_p = false;
            for (std::size_t k = loop.open + 1; k < result.size(); ++k) {
                const Op op = result[k].op;
                if (op == Op::Move || op == Op::Scan || op == Op::MoveJnz) {
                    body_modified_p = true;
                    break;
                }
            }

            if (!body_modified_p) {
                if (const int shift = pending - loop.base; shift == 0) {
                    if (try_optimize_balanced(result, loop.open, loop.base)) {
                        continue;
                    }
                    result.push_back({.op = Op::JumpIfNotZero, .arg = static_cast<int>(loop.open), .off = loop.base});
                    result[loop.open].arg = static_cast<int>(result.size() - 1);
                    pending = loop.base;
                } else if (result.size() == loop.open + 1) {
                    // Pure scan loop: [>], [<], etc.
                    result.pop_back(); // remove JumpIfZero
                    // First rebase any outer loops that still had non-zero base
                    for (std::size_t s = 0; s < stack.size(); ++s) {
                        if (stack[s].base != 0) {
                            const int b = stack[s].base;
                            const std::size_t old_len = result.size();
                            rebase_loop_to_zero(result, stack[s]);
                            const int len_diff = static_cast<int>(result.size()) - static_cast<int>(old_len);
                            for (std::size_t inner = s + 1; inner < stack.size(); ++inner) {
                                stack[inner].base -= b;
                                stack[inner].open = static_cast<std::size_t>(
                                    static_cast<int>(stack[inner].open) + len_diff);
                            }
                            loop.base -= b;
                        }
                    }
                    emit_move(loop.base);
                    result.push_back({.op = Op::Scan, .arg = shift});
                    has_moved = true;
                    pending = 0;
                } else {
                    // Unbalanced loop without inner Moves yet: rebase all outer loops + this loop to 0
                    for (std::size_t s = 0; s < stack.size(); ++s) {
                        if (stack[s].base != 0) {
                            const int b = stack[s].base;
                            const std::size_t old_len = result.size();
                            rebase_loop_to_zero(result, stack[s]);
                            const int len_diff = static_cast<int>(result.size()) - static_cast<int>(old_len);
                            for (std::size_t inner = s + 1; inner < stack.size(); ++inner) {
                                stack[inner].base -= b;
                                stack[inner].open = static_cast<std::size_t>(
                                    static_cast<int>(stack[inner].open) + len_diff);
                            }
                            loop.base -= b;
                            loop.open = static_cast<std::size_t>(static_cast<int>(loop.open) + len_diff);
                        }
                    }
                    rebase_loop_to_zero(result, loop);
                    result.push_back({.op = Op::MoveJnz, .arg = static_cast<int>(loop.open), .off = 0, .off2 = shift});
                    result[loop.open].arg = static_cast<int>(result.size() - 1);
                    has_moved = true;
                    pending = 0;
                }
            } else {
                // `body_modified_p == true`: Because any inner Scan/MoveJnz ALREADY rebased all its enclosing
                // loops (including `loop`) to `base == 0`, `loop.base` is guaranteed to be 0 here!
                if (pending != 0) {
                    result.push_back({
                        .op = Op::MoveJnz, .arg = static_cast<int>(loop.open), .off = 0, .off2 = pending
                    });
                    pending = 0;
                } else {
                    result.push_back({.op = Op::JumpIfNotZero, .arg = static_cast<int>(loop.open), .off = 0});
                }
                result[loop.open].arg = static_cast<int>(result.size() - 1);
                has_moved = true;
            }
        }
    }
    if (!stack.empty()) throw std::runtime_error{"Unclosed ["};
    result.push_back({.op = Op::Halt, .arg = 0});


    return pre_evaluate(result);
}

// ---------------------------------------------------------------------------
// Backend: Multi-level SSA Register Promotion + Apple Silicon NEON C Codegen
// ---------------------------------------------------------------------------
// Tunable code-generation / compiler-flag knobs; `--tune` picks the fastest combination for a given program.
struct Config {
    int scan{3};
    // 0 = plain scalar, 1 = NEON for stride +-1 only, 2 = NEON for all strides (slow), 3 = NEON +-1, unrolled multi-exit scalar otherwise
    int unroll{4}; // unroll factor of the multi-exit scalar scans (scan == 3)
    int flagset{0}; // extra clang flags, see config_flags()
    int promote{28}; // max cells held in registers for one promoted loop
    int strided{1}; // carry loop-carried cells of strided record loops in registers
    int pgo{0}; // build with a profile trained on this program (see train_pgo)
    int branchless{0}; // 0 = off, 1 = if-convert `if (cell)` bodies up to 4 ops, 2 = up to 12 ops
};

static Config g_cfg;

static std::string config_tag(const Config &c) {
    return std::format("s{}f{}p{}b{}u{}r{}g{}", c.scan, c.flagset, c.promote, c.branchless, c.unroll, c.strided, c.pgo);
}

static std::string config_flags(const int flagset) {
    switch (flagset) {
        case 1: return "-fno-slp-vectorize -fno-vectorize";
        case 2: return "-fno-unroll-loops";
        case 3: return "-fno-slp-vectorize -fno-vectorize -fno-unroll-loops";
        case 4: return "-O2";
        default: return "";
    }
}

static std::string replace_all(std::string s, const std::string_view from, const std::string &to) {
    for (std::size_t pos = 0; (pos = s.find(from, pos)) != std::string::npos; pos += to.size()) {
        s.replace(pos, from.size(), to);
    }
    return s;
}

// Emits NEON scan helpers for strides 3..16 (except 4) that actually occur in the program.
static void emit_generic_scans(std::string &out, const std::vector<Ir> &ir) {
    static constexpr std::string_view TEMPLATE = R"BF(#ifdef __ARM_NEON
static const uint64_t bf_pat_@S@[@S@] = {@T@};
static inline __attribute__((always_inline)) u8* scan_right_@S@(u8 *p) {
    BF_PRE_R(@S@)
    if (*p == 0) return p;
    const unsigned mis = (unsigned)((uintptr_t)p & 15);
    u8 *b = p - mis;
    unsigned ph = mis % @S@u;
    uint64_t m = neon_zero_mask(vld1q_u8(b)) & bf_pat_@S@[ph] & (~0ULL << (4 * mis));
    for (;;) {
        if (m) return b + (__builtin_ctzll(m) >> 2);
        b += 16;
        ph = ph >= @R@u ? ph - @R@u : ph + @S@u - @R@u;
        m = neon_zero_mask(vld1q_u8(b)) & bf_pat_@S@[ph];
    }
}
static inline __attribute__((always_inline)) u8* scan_left_@S@(u8 *p) {
    BF_PRE_L(@S@)
    if (*p == 0) return p;
    const unsigned mis = (unsigned)((uintptr_t)p & 15);
    u8 *b = p - mis;
    unsigned ph = mis % @S@u;
    uint64_t m = neon_zero_mask(vld1q_u8(b)) & bf_pat_@S@[ph] & (~0ULL >> (60 - 4 * mis));
    for (;;) {
        if (m) return b + 15 - (__builtin_clzll(m) >> 2);
        b -= 16;
        ph += @R@u;
        if (ph >= @S@u) ph -= @S@u;
        m = neon_zero_mask(vld1q_u8(b)) & bf_pat_@S@[ph];
    }
}
#else
static inline u8* scan_right_@S@(u8 *p) { while (*p) p += @S@; return p; }
static inline u8* scan_left_@S@(u8 *p) { while (*p) p -= @S@; return p; }
#endif
)BF";

    std::set<int> strides;
    for (const Ir &in: ir) {
        if (in.op != Op::Scan) continue;
        if (const int a = std::abs(in.arg); a >= 3 && a <= 16 && a != 4) strides.insert(a);
    }
    for (const int s: strides) {
        std::string tbl;
        for (int ph = 0; ph < s; ++ph) {
            std::uint64_t pat = 0;
            for (int lane = ph; lane < 16; lane += s) pat |= 0xFULL << (4 * lane);
            tbl += std::format("0x{:016x}ULL,", pat);
        }
        std::string t = replace_all(std::string{TEMPLATE}, "@S@", std::to_string(s));
        t = replace_all(std::move(t), "@R@", std::to_string(16 % s));
        t = replace_all(std::move(t), "@T@", tbl);
        out += t;
        out += '\n';
    }
}

// Multi-exit unrolled scalar scans for strides other than +-1: every probe has its own branch, which the
// predictor learns per position, and the pointer is bumped once per `unroll` probes.
static void emit_unrolled_scans(std::string &out, const std::vector<Ir> &ir) {
    std::set<int> strides;
    for (const Ir &in: ir) {
        if (in.op == Op::Scan && std::abs(in.arg) >= 2) strides.insert(in.arg);
    }
    for (const int s: strides) {
        const int a = std::abs(s);
        std::string body;
        for (int k = 0; k < g_cfg.unroll; ++k) {
            const int d = k * s;
            if (d == 0) body += "if (!p[0]) return p; ";
            else body += std::format("if (!p[{}]) return p + {}; ", d, d);
        }
        out += std::format(
            "static inline __attribute__((always_inline)) u8* bf_scan_{}{}(u8 *p) {{ for (;;) {{ {}p += {}; }} }}\n",
            s > 0 ? "p" : "n", a, body, g_cfg.unroll * s);
    }
    out += '\n';
}

static std::string generate_c(const ProgramIr &prog) {
    const auto &ir = prog.code;
    std::string out;
    out.reserve(ir.size() * 32 + prog.pre_output.size() * 4 + 4096);

    out +=
            "#include <stdint.h>\n"
            "#include <stdio.h>\n"
            "#include <string.h>\n"
            "#include <unistd.h>\n"
            "#ifdef __ARM_NEON\n"
            "#include <arm_neon.h>\n"
            "#endif\n\n"
            "typedef uint8_t u8;\n"
            "#define BF_PRE_R(N) if (*p == 0) return p; p += (N); if (*p == 0) return p; p += (N); if (*p == 0) return p; p += (N);\n"
            "#define BF_PRE_L(N) if (*p == 0) return p; p -= (N); if (*p == 0) return p; p -= (N); if (*p == 0) return p; p -= (N);\n\n"
            "#ifdef __ARM_NEON\n"
            "static inline __attribute__((always_inline)) uint64_t neon_zero_mask(uint8x16_t v) {\n"
            "    uint8x16_t cmp = vceqq_u8(v, vdupq_n_u8(0));\n"
            "    return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(cmp), 4)), 0);\n"
            "}\n\n"
            "static inline __attribute__((always_inline)) u8* scan_right_1(u8 *p) {\n"
            "    if (*p == 0) return p;\n"
            "    uintptr_t mis = (uintptr_t)p & 15;\n"
            "    if (mis != 0) {\n"
            "        uintptr_t step = 16 - mis;\n"
            "        for (uintptr_t i = 0; i < step; ++i) {\n"
            "            if (p[i] == 0) return p + i;\n"
            "        }\n"
            "        p += step;\n"
            "    }\n"
            "    for (;;) {\n"
            "        uint64_t mask = neon_zero_mask(vld1q_u8(p));\n"
            "        if (mask != 0) return p + (__builtin_ctzll(mask) >> 2);\n"
            "        p += 16;\n"
            "    }\n"
            "}\n\n"
            "static inline __attribute__((always_inline)) u8* scan_left_1(u8 *p) {\n"
            "    for (;;) {\n"
            "        if (*p == 0) return p;\n"
            "        if (((uintptr_t)p & 15) == 0) break;\n"
            "        --p;\n"
            "    }\n"
            "    for (;;) {\n"
            "        p -= 16;\n"
            "        uint64_t mask = neon_zero_mask(vld1q_u8(p));\n"
            "        if (mask != 0) return p + 15 - (__builtin_clzll(mask) >> 2);\n"
            "    }\n"
            "}\n"
            "static inline __attribute__((always_inline)) u8* scan_right_2(u8 *p) {\n"
            "    BF_PRE_R(2)\n"
            "    if (*p == 0) return p;\n"
            "    while (((uintptr_t)p & 30) != 0) { p += 2; if (*p == 0) return p; }\n"
            "    const unsigned par = (unsigned)((uintptr_t)p & 1);\n"
            "    p -= par;\n"
            "    for (;;) {\n"
            "        uint8x16x2_t v = vld2q_u8(p);\n"
            "        uint8x16_t l = v.val[0];\n"
            "        if (par) l = v.val[1];\n"
            "        uint64_t m = neon_zero_mask(l);\n"
            "        if (m) return p + par + ((unsigned)__builtin_ctzll(m) >> 2) * 2;\n"
            "        p += 32;\n"
            "    }\n"
            "}\n\n"
            "static inline __attribute__((always_inline)) u8* scan_left_2(u8 *p) {\n"
            "    BF_PRE_L(2)\n"
            "    for (;;) {\n"
            "        if (*p == 0) return p;\n"
            "        if (((uintptr_t)p & 31) < 2) break;\n"
            "        p -= 2;\n"
            "    }\n"
            "    const unsigned par = (unsigned)((uintptr_t)p & 1);\n"
            "    u8 *b = p - par - 32;\n"
            "    for (;;) {\n"
            "        uint8x16x2_t v = vld2q_u8(b);\n"
            "        uint8x16_t l = v.val[0];\n"
            "        if (par) l = v.val[1];\n"
            "        uint64_t m = neon_zero_mask(l);\n"
            "        if (m) return b + par + (15 - ((unsigned)__builtin_clzll(m) >> 2)) * 2;\n"
            "        b -= 32;\n"
            "    }\n"
            "}\n\n"
            "static inline __attribute__((always_inline)) u8* scan_right_4(u8 *p) {\n"
            "    BF_PRE_R(4)\n"
            "    if (*p == 0) return p;\n"
            "    while (((uintptr_t)p & 60) != 0) { p += 4; if (*p == 0) return p; }\n"
            "    const unsigned par = (unsigned)((uintptr_t)p & 3);\n"
            "    p -= par;\n"
            "    for (;;) {\n"
            "        uint8x16x4_t v = vld4q_u8(p);\n"
            "        uint8x16_t l = v.val[0];\n"
            "        if (par == 1) l = v.val[1];\n"
            "        else if (par == 2) l = v.val[2];\n"
            "        else if (par == 3) l = v.val[3];\n"
            "        uint64_t m = neon_zero_mask(l);\n"
            "        if (m) return p + par + ((unsigned)__builtin_ctzll(m) >> 2) * 4;\n"
            "        p += 64;\n"
            "    }\n"
            "}\n\n"
            "static inline __attribute__((always_inline)) u8* scan_left_4(u8 *p) {\n"
            "    BF_PRE_L(4)\n"
            "    for (;;) {\n"
            "        if (*p == 0) return p;\n"
            "        if (((uintptr_t)p & 63) < 4) break;\n"
            "        p -= 4;\n"
            "    }\n"
            "    const unsigned par = (unsigned)((uintptr_t)p & 3);\n"
            "    u8 *b = p - par - 64;\n"
            "    for (;;) {\n"
            "        uint8x16x4_t v = vld4q_u8(b);\n"
            "        uint8x16_t l = v.val[0];\n"
            "        if (par == 1) l = v.val[1];\n"
            "        else if (par == 2) l = v.val[2];\n"
            "        else if (par == 3) l = v.val[3];\n"
            "        uint64_t m = neon_zero_mask(l);\n"
            "        if (m) return b + par + (15 - ((unsigned)__builtin_clzll(m) >> 2)) * 4;\n"
            "        b -= 64;\n"
            "    }\n"
            "}\n"
            "#else\n"
            "static inline u8* scan_right_1(u8 *p) { while (*p) ++p; return p; }\n"
            "static inline u8* scan_left_1(u8 *p) { while (*p) --p; return p; }\n"
            "static inline u8* scan_right_2(u8 *p) { while (*p) p += 2; return p; }\n"
            "static inline u8* scan_left_2(u8 *p) { while (*p) p -= 2; return p; }\n"
            "static inline u8* scan_right_4(u8 *p) { while (*p) p += 4; return p; }\n"
            "static inline u8* scan_left_4(u8 *p) { while (*p) p -= 4; return p; }\n"
            "#endif\n\n";
    if (g_cfg.scan == 2) emit_generic_scans(out, ir);
    if (g_cfg.scan == 3) emit_unrolled_scans(out, ir);
    out +=
            "static u8 out_buf[65536];\n"
            "static size_t out_pos;\n"
            "static int bf_lb;\n\n"
            "__attribute__((noinline)) static size_t bf_flush(u8 *buf, size_t pos) {\n"
            "    if (pos > 0) (void)write(STDOUT_FILENO, buf, pos);\n"
            "    return 0;\n"
            "}\n\n"
            "static inline __attribute__((always_inline)) void bf_out(u8 c, u8 *out_buf, size_t *out_pos) {\n"
            "    out_buf[(*out_pos)++] = c;\n"
            "    if (__builtin_expect(*out_pos == 65536 || (c == 10 && bf_lb), 0)) {\n"
            "        *out_pos = bf_flush(out_buf, *out_pos);\n"
            "    }\n"
            "}\n\n"
            "static const u8 *bf_in;\n"
            "static size_t bf_in_len, bf_in_pos;\n"
            "static int bf_in_set;\n"
            "void bf_set_input(const u8 *p, size_t n) { bf_in = p; bf_in_len = n; bf_in_pos = 0; bf_in_set = 1; }\n\n"
            "__attribute__((noinline)) static u8 bf_get(u8 *out_buf, size_t *pos) {\n"
            "    if (bf_in_set) return bf_in_pos < bf_in_len ? bf_in[bf_in_pos++] : 0;\n"
            "    *pos = bf_flush(out_buf, *pos);\n"
            "    int c = getchar_unlocked();\n"
            "    return (u8)(c == EOF ? 0 : c);\n"
            "}\n\n"
            "#ifdef BF_PGO\n"
            "extern int __llvm_profile_write_file(void);\n"
            "void bf_pgo_dump(void) { (void)__llvm_profile_write_file(); }\n"
            "#endif\n\n"
            "void bf_flush_out(void) { out_pos = bf_flush(out_buf, out_pos); }\n\n"
            "void bf_run(u8 * __restrict base) {\n"
            "    u8 * __restrict p = base;\n"
            "    bf_lb = isatty(STDOUT_FILENO);\n";

    int depth = 1;
    const auto line = [&](const std::string_view s) {
        out.append(static_cast<std::size_t>(depth) * 2 + 2, ' ');
        out.append(s);
        out += '\n';
    };

    if (!prog.pre_output.empty()) {
        std::string hex_lit;
        hex_lit.reserve(prog.pre_output.size() * 4);
        for (const char ch: prog.pre_output) {
            hex_lit += std::format("\\x{:02x}", static_cast<unsigned char>(ch));
        }
        line(std::format("(void)write(STDOUT_FILENO, \"{}\", {});", hex_lit, prog.pre_output.size()));
    }

    const auto make_reg = [](const int off) -> std::string {
        return (off >= 0) ? std::format("r_p{}", off) : std::format("r_m{}", -off);
    };

    // If-conversion of `if (cell) { straight-line Add/Set/MulAdd }`: run the body on temporaries and select with csel,
    // so unpredictable data-dependent conditions cost no branch mispredict. Returns the EndIf index, or 0 if not applied.
    const auto try_branchless = [&](const std::size_t open, const bool regs) -> std::size_t {
        if (g_cfg.branchless == 0) return 0;
        const auto end_idx = static_cast<std::size_t>(ir[open].arg);
        const std::size_t max_ops = g_cfg.branchless == 1 ? 4 : 12;
        if (end_idx <= open + 1 || end_idx - open - 1 > max_ops) return 0;

        std::set<int> cells;
        std::set<int> written;
        cells.insert(ir[open].off);
        for (std::size_t k = open + 1; k < end_idx; ++k) {
            const Ir &b = ir[k];
            switch (b.op) {
                case Op::Add:
                case Op::Set:
                    cells.insert(b.off);
                    written.insert(b.off);
                    break;
                case Op::MulAdd:
                case Op::MulAddClear:
                    cells.insert(b.off);
                    cells.insert(b.off2);
                    written.insert(b.off2);
                    if (b.op == Op::MulAddClear) written.insert(b.off);
                    break;
                default:
                    return 0;
            }
        }

        const auto ref = [&](const int off) -> std::string {
            return regs ? make_reg(off) : std::format("p[{}]", off);
        };
        const auto tmp = [](const int off) -> std::string {
            return off >= 0 ? std::format("t_p{}", off) : std::format("t_m{}", -off);
        };

        line("{");
        ++depth;
        line(std::format("const int cnd = {} != 0;", ref(ir[open].off)));
        for (const int off: cells) line(std::format("u8 {} = {};", tmp(off), ref(off)));
        for (std::size_t k = open + 1; k < end_idx; ++k) {
            const Ir &b = ir[k];
            switch (b.op) {
                case Op::Add:
                    line(std::format("{} += {}u;", tmp(b.off), mod256(b.arg)));
                    break;
                case Op::Set:
                    line(std::format("{} = {}u;", tmp(b.off), mod256(b.arg)));
                    break;
                default:
                    if (b.arg == 1) {
                        line(std::format("{} += {};", tmp(b.off2), tmp(b.off)));
                    } else if (b.arg == 255) {
                        line(std::format("{} -= {};", tmp(b.off2), tmp(b.off)));
                    } else {
                        line(std::format("{} += (u8)({}u * {});", tmp(b.off2), b.arg, tmp(b.off)));
                    }
                    if (b.op == Op::MulAddClear) line(std::format("{} = 0;", tmp(b.off)));
                    break;
            }
        }
        for (const int off: written) line(std::format("{} = cnd ? {} : {};", ref(off), tmp(off), ref(off)));
        --depth;
        line("}");
        return end_idx;
    };

    // Strided record loops `while (p[t]) { straight-line body; p += s; }`: keep the body's cells in registers and carry
    // values that the next iteration reads at a shifted offset (written at o, read at o - s) in a register instead of
    // going through memory, which removes the store->load forwarding latency from the loop-carried chain.
    const auto try_strided = [&](const std::size_t open) -> std::size_t {
        if (!g_cfg.strided) return 0;
        const auto end_idx = static_cast<std::size_t>(ir[open].arg);
        if (ir[end_idx].op != Op::MoveJnz) return 0;
        const int s = ir[end_idx].off2;
        if (s == 0) return 0;

        std::set<int> reads;
        std::set<int> writes;
        reads.insert(ir[open].off);
        for (std::size_t k = open + 1; k < end_idx; ++k) {
            const Ir &b = ir[k];
            switch (b.op) {
                case Op::Add:
                    reads.insert(b.off);
                    writes.insert(b.off);
                    break;
                case Op::Set:
                    writes.insert(b.off);
                    break;
                case Op::MulAdd:
                case Op::MulAddClear:
                    reads.insert(b.off);
                    reads.insert(b.off2);
                    writes.insert(b.off2);
                    if (b.op == Op::MulAddClear) writes.insert(b.off);
                    break;
                case Op::MulCells:
                    reads.insert(b.off);
                    reads.insert(b.off2);
                    reads.insert(b.off3);
                    writes.insert(b.off2);
                    break;
                default:
                    return 0;
            }
        }
        std::set<int> all = reads;
        all.insert(writes.begin(), writes.end());
        if (all.size() > static_cast<std::size_t>(g_cfg.promote)) return 0;

        std::set<int> carried; // offsets whose value arrives from the previous iteration's register
        for (const int o: writes) {
            if (reads.contains(o - s)) carried.insert(o - s);
        }

        line("{");
        ++depth;
        for (const int o: all) line(std::format("u8 {} = 0;", make_reg(o)));
        for (const int o: carried) line(std::format("{} = p[{}];", make_reg(o), o));
        line("for (;;) {");
        ++depth;
        for (const int o: reads) {
            if (!carried.contains(o)) line(std::format("{} = p[{}];", make_reg(o), o));
        }
        line(std::format("if (!{}) break;", make_reg(ir[open].off)));
        for (std::size_t k = open + 1; k < end_idx; ++k) {
            const Ir &b = ir[k];
            switch (b.op) {
                case Op::Add:
                    line(std::format("{} += {}u;", make_reg(b.off), mod256(b.arg)));
                    break;
                case Op::Set:
                    line(std::format("{} = {}u;", make_reg(b.off), mod256(b.arg)));
                    break;
                case Op::MulCells:
                    line(std::format("{} += (u8)({}u * {} * {});", make_reg(b.off2), b.arg, make_reg(b.off),
                                     make_reg(b.off3)));
                    break;
                default:
                    if (b.arg == 1) {
                        line(std::format("{} += {};", make_reg(b.off2), make_reg(b.off)));
                    } else if (b.arg == 255) {
                        line(std::format("{} -= {};", make_reg(b.off2), make_reg(b.off)));
                    } else {
                        line(std::format("{} += (u8)({}u * {});", make_reg(b.off2), b.arg, make_reg(b.off)));
                    }
                    if (b.op == Op::MulAddClear) line(std::format("{} = 0;", make_reg(b.off)));
                    break;
            }
        }
        for (const int o: writes) line(std::format("p[{}] = {};", o, make_reg(o)));
        std::vector<int> order(writes.begin(), writes.end());
        if (s < 0) std::ranges::reverse(order);
        for (const int o: order) {
            if (carried.contains(o - s)) line(std::format("{} = {};", make_reg(o - s), make_reg(o)));
        }
        line(std::format("p += {};", s));
        --depth;
        line("}");
        --depth;
        line("}");
        return end_idx;
    };

    for (std::size_t i = 0; i < ir.size(); ++i) {
        const Ir &in = ir[i];
        switch (in.op) {
            case Op::Add:
                line(std::format("p[{}] += {}u;", in.off, mod256(in.arg)));
                break;
            case Op::Set:
                line(std::format("p[{}] = {}u;", in.off, mod256(in.arg)));
                break;
            case Op::Move:
                line(std::format("p += {};", in.arg));
                break;
            case Op::Out:
                line(std::format("bf_out(p[{}], out_buf, &out_pos);", in.off));
                break;
            case Op::In:
                line(std::format("p[{}] = bf_get(out_buf, &out_pos);", in.off));
                break;
            case Op::IfNonZero:
                if (const std::size_t e = try_branchless(i, false); e != 0) {
                    i = e;
                    break;
                }
                line(std::format("if (p[{}]) {{", in.off));
                ++depth;
                break;
            case Op::EndIf:
                --depth;
                line("}");
                break;
            case Op::JumpIfZero: {
                if (const std::size_t e = try_strided(i); e != 0) {
                    i = e;
                    break;
                }
                std::set<int> read_cells;
                std::set<int> written_cells;
                read_cells.insert(in.off);

                bool can_promote = true;
                const auto close_idx = static_cast<std::size_t>(in.arg);

                for (std::size_t k = i + 1; k <= close_idx; ++k) {
                    const Ir &body = ir[k];
                    switch (body.op) {
                        case Op::Add:
                            read_cells.insert(body.off);
                            written_cells.insert(body.off);
                            break;
                        case Op::Set:
                            written_cells.insert(body.off);
                            break;
                        case Op::MulAdd:
                        case Op::MulAddClear:
                            read_cells.insert(body.off);
                            read_cells.insert(body.off2);
                            written_cells.insert(body.off2);
                            if (body.op == Op::MulAddClear) written_cells.insert(body.off);
                            break;
                        case Op::MulCells:
                            read_cells.insert(body.off);
                            read_cells.insert(body.off2);
                            read_cells.insert(body.off3);
                            written_cells.insert(body.off2);
                            break;
                        case Op::JumpIfZero:
                        case Op::JumpIfNotZero:
                        case Op::IfNonZero:
                        case Op::EndIf:
                        case Op::Out:
                            read_cells.insert(body.off);
                            break;
                        default:
                            can_promote = false;
                            break;
                    }
                    if (!can_promote) break;
                }

                std::set<int> all_cells = read_cells;
                all_cells.insert(written_cells.begin(), written_cells.end());

                if (can_promote && all_cells.size() <= static_cast<std::size_t>(g_cfg.promote)) {
                    line("{");
                    ++depth;
                    for (const int off: all_cells) {
                        line(std::format("u8 {} = p[{}];", make_reg(off), off));
                    }
                    line(std::format("while ({}) {{", make_reg(in.off)));
                    ++depth;

                    for (std::size_t k = i + 1; k < close_idx; ++k) {
                        const Ir &body = ir[k];
                        switch (body.op) {
                            case Op::Add:
                                line(std::format("{} += {}u;", make_reg(body.off), mod256(body.arg)));
                                break;
                            case Op::Set:
                                line(std::format("{} = {}u;", make_reg(body.off), mod256(body.arg)));
                                break;
                            case Op::MulAdd:
                                if (body.arg == 1) {
                                    line(std::format("{} += {};", make_reg(body.off2), make_reg(body.off)));
                                } else if (body.arg == 255) {
                                    line(std::format("{} -= {};", make_reg(body.off2), make_reg(body.off)));
                                } else {
                                    line(std::format("{} += (u8)({}u * {});", make_reg(body.off2), body.arg,
                                                     make_reg(body.off)));
                                }
                                break;
                            case Op::MulAddClear:
                                if (body.arg == 1) {
                                    line(std::format("{} += {};", make_reg(body.off2), make_reg(body.off)));
                                } else if (body.arg == 255) {
                                    line(std::format("{} -= {};", make_reg(body.off2), make_reg(body.off)));
                                } else {
                                    line(std::format("{} += (u8)({}u * {});", make_reg(body.off2), body.arg,
                                                     make_reg(body.off)));
                                }
                                line(std::format("{} = 0;", make_reg(body.off)));
                                break;
                            case Op::MulCells:
                                line(std::format("{} += (u8)({}u * {} * {});", make_reg(body.off2), body.arg,
                                                 make_reg(body.off), make_reg(body.off3)));
                                break;
                            case Op::JumpIfZero:
                                line(std::format("while ({}) {{", make_reg(body.off)));
                                ++depth;
                                break;
                            case Op::JumpIfNotZero:
                                --depth;
                                line("}");
                                break;
                            case Op::IfNonZero:
                                if (const std::size_t e = try_branchless(k, true); e != 0) {
                                    k = e;
                                    break;
                                }
                                line(std::format("if ({}) {{", make_reg(body.off)));
                                ++depth;
                                break;
                            case Op::EndIf:
                                --depth;
                                line("}");
                                break;
                            case Op::Out:
                                line(std::format("bf_out({}, out_buf, &out_pos);", make_reg(body.off)));
                                break;
                            default:
                                break;
                        }
                    }

                    --depth;
                    line("}");
                    for (const int off: written_cells) {
                        line(std::format("p[{}] = {};", off, make_reg(off)));
                    }
                    --depth;
                    line("}");
                    i = close_idx;
                } else {
                    line(std::format("while (p[{}]) {{", in.off));
                    ++depth;
                }
                break;
            }
            case Op::JumpIfNotZero:
                --depth;
                line("}");
                break;
            case Op::MoveJnz:
                line(std::format("p += {};", in.off2));
                --depth;
                line("}");
                break;
            case Op::MulAdd:
                if (in.arg == 1) {
                    line(std::format("p[{}] += p[{}];", in.off2, in.off));
                } else if (in.arg == 255) {
                    line(std::format("p[{}] -= p[{}];", in.off2, in.off));
                } else {
                    line(std::format("p[{}] += (u8)({}u * p[{}]);", in.off2, in.arg, in.off));
                }
                break;
            case Op::MulAddClear:
                if (in.arg == 1) {
                    line(std::format("p[{}] += p[{}];", in.off2, in.off));
                } else if (in.arg == 255) {
                    line(std::format("p[{}] -= p[{}];", in.off2, in.off));
                } else {
                    line(std::format("p[{}] += (u8)({}u * p[{}]);", in.off2, in.arg, in.off));
                }
                line(std::format("p[{}] = 0;", in.off));
                break;
            case Op::MulCells:
                line(std::format("p[{}] += (u8)({}u * p[{}] * p[{}]);", in.off2, in.arg, in.off, in.off3));
                break;
            case Op::Scan:
                if (g_cfg.scan == 3 && std::abs(in.arg) >= 2) {
                    line(std::format("p = bf_scan_{}{}(p);", in.arg > 0 ? "p" : "n", std::abs(in.arg)));
                    break;
                }
                if (g_cfg.scan == 0 || (g_cfg.scan == 1 && std::abs(in.arg) != 1)) {
                    line(std::format("while (*p) p += {};", in.arg));
                    break;
                }
                if (in.arg == 1) {
                    line("p = scan_right_1(p);");
                } else if (in.arg == -1) {
                    line("p = scan_left_1(p);");
                } else if (in.arg == 2) {
                    line("p = scan_right_2(p);");
                } else if (in.arg == -2) {
                    line("p = scan_left_2(p);");
                } else if (in.arg == 4) {
                    line("p = scan_right_4(p);");
                } else if (in.arg == -4) {
                    line("p = scan_left_4(p);");
                } else if (std::abs(in.arg) >= 3 && std::abs(in.arg) <= 16) {
                    line(std::format("p = scan_{}_{}(p);", in.arg > 0 ? "right" : "left", std::abs(in.arg)));
                } else {
                    line(std::format("while (*p) p += {};", in.arg));
                }
                break;
            case Op::Halt:
                break;
        }
    }
    line("(void)bf_flush(out_buf, out_pos);");
    out += "}\n";
    return out;
}

// ---------------------------------------------------------------------------
// Compiler Invocation & Execution
// ---------------------------------------------------------------------------
static std::uint64_t fnv1a(const std::string_view s) noexcept {
    std::uint64_t h = 1469598103934665603ULL;
    for (const char ch: s) {
        h ^= static_cast<unsigned char>(ch);
        h *= 1099511628211ULL;
    }
    return h;
}

static std::string shell_quote(const std::string_view s) {
    std::string res = "'";
    for (const char c: s) {
        if (c == '\'') res += "'\\''";
        else res += c;
    }
    res += '\'';
    return res;
}

namespace fs = std::filesystem;

static fs::path resolve_cc(const std::string &cc) {
    const std::string exe = cc.substr(0, cc.find(' '));
    fs::path found;
    if (exe.find('/') != std::string::npos) {
        found = exe;
    } else if (const char *path_env = std::getenv("PATH")) {
        const std::string_view paths{path_env};
        for (std::size_t start = 0; start <= paths.size();) {
            std::size_t end = paths.find(':', start);
            if (end == std::string_view::npos) end = paths.size();
            const fs::path cand = fs::path{std::string{paths.substr(start, end - start)}} / exe;
            if (::access(cand.c_str(), X_OK) == 0) {
                found = cand;
                break;
            }
            start = end + 1;
        }
    }
    if (found.empty()) return {};
    std::error_code ec;
    const fs::path real = fs::canonical(found, ec);
    return ec ? fs::path{} : real;
}

struct Toolchain {
    std::string cc;
    std::string flags;
    fs::path dir;
    const char *ext;
    std::string cc_id; // resolved compiler path, so switching toolchains invalidates the cache
};

static Toolchain get_toolchain() {
#ifdef __APPLE__
    const char *ext = ".dylib";
    const char *default_flags =
            "-O3 -mcpu=native -fomit-frame-pointer -fno-stack-protector -fstrict-aliasing -fvectorize -fslp-vectorize "
            "-fno-unwind-tables -fno-asynchronous-unwind-tables";
#else
    const char *ext = ".so";
    const char *default_flags = "-O3 -march=native -fomit-frame-pointer -fno-stack-protector -fstrict-aliasing";
#endif
    const char *cc_env = std::getenv("BF_CC");
    const char *flags_env = std::getenv("BF_CFLAGS");
    const char *dir_env = std::getenv("BF_CACHE_DIR");

    const std::string cc = cc_env ? cc_env : "clang";
    const fs::path cc_real = resolve_cc(cc);
    Toolchain t{
        .cc = cc,
        .flags = std::string{flags_env ? flags_env : default_flags} + " -fno-finite-loops",
        .dir = dir_env ? fs::path{dir_env} : fs::temp_directory_path() / "bf_native_cache",
        .ext = ext,
        .cc_id = cc_real.empty() ? cc : cc_real.string()
    };
    fs::create_directories(t.dir);
    return t;
}

static fs::path pgo_profile_path(const std::string &program, const Toolchain &tc);

struct BuildPlan {
    std::string lib; // final library path
    bool cached{false};
    std::string cmd; // compile command, writes to `tmp`
    fs::path src;
    fs::path tmp;
};

// Computes the cache key/paths. With `generate` the C source is written and the compile command is prepared.
static BuildPlan plan_build(const std::string &program, const bool no_cache, const std::string &extra = {},
                            const std::string &tag = {}, const ProgramIr *prebuilt = nullptr,
                            const bool generate = true) {
    const Toolchain tc = get_toolchain();
    std::string pgo_flags;
    if (g_cfg.pgo) {
        const fs::path prof = pgo_profile_path(program, tc);
        if (!fs::exists(prof)) throw std::runtime_error{"PGO profile missing: " + prof.string()};
        pgo_flags = " -fprofile-use=" + shell_quote(prof.string());
    }
    const std::string all_extra = extra + " " + config_flags(g_cfg.flagset) + pgo_flags;
    const std::string key = std::format(
        "{:016x}", fnv1a(program + '\n' + tc.cc_id + '\n' + tc.flags + '\n' + all_extra + '\n' + tag + '\n' +
                         config_tag(g_cfg) + "\ncodegen-v12"));
    const fs::path lib = tc.dir / ("bf_" + key + tc.ext);

    BuildPlan plan;
    plan.lib = lib.string();
    if (!no_cache && fs::exists(lib)) {
        plan.cached = true;
        return plan;
    }
    if (!generate) return plan;

    plan.tmp = tc.dir / std::format("bf_{}_{}.tmp{}", key, ::getpid(), tc.ext);
    plan.src = tc.dir / std::format("bf_{}_{}.c", key, ::getpid());
    {
        const std::string c_source = generate_c(prebuilt ? *prebuilt : convert(program));
        std::ofstream f{plan.src, std::ios::binary | std::ios::trunc};
        f.write(c_source.data(), static_cast<std::streamsize>(c_source.size()));
        f.flush();
        if (!f) throw std::runtime_error{"Failed to write temporary C source"};
    }
    plan.cmd = tc.cc + " " + tc.flags + " " + all_extra + " -fPIC -shared -w -x c " + shell_quote(plan.src.string()) +
               " -o " + shell_quote(plan.tmp.string());
    return plan;
}

static std::string build_library(const std::string &program, const bool no_cache,
                                 const std::string &extra = {}, const std::string &tag = {}) {
    const BuildPlan plan = plan_build(program, no_cache, extra, tag);
    if (plan.cached) return plan.lib;
    const int rc = std::system(plan.cmd.c_str());
    std::error_code ec;
    fs::remove(plan.src, ec);
    if (rc != 0) {
        fs::remove(plan.tmp, ec);
        throw std::runtime_error{"Compilation failed (command: " + plan.cmd + ")"};
    }
    fs::rename(plan.tmp, plan.lib);
    return plan.lib;
}

using RunFn = void (*)(unsigned char *);

struct LibraryHandle {
    void *handle{nullptr};
    RunFn fn{nullptr};

    void (*flush)(){nullptr};

    void (*dump)(){nullptr};

    void (*set_input)(const unsigned char *, std::size_t){nullptr};

    explicit LibraryHandle(const std::string &path) {
        handle = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) throw std::runtime_error{std::string{"dlopen failed: "} + ::dlerror()};
        void *sym = ::dlsym(handle, "bf_run");
        if (!sym) {
            ::dlclose(handle);
            throw std::runtime_error{"bf_run symbol not found"};
        }
        fn = reinterpret_cast<RunFn>(sym);
        flush = reinterpret_cast<void (*)()>(::dlsym(handle, "bf_flush_out"));
        dump = reinterpret_cast<void (*)()>(::dlsym(handle, "bf_pgo_dump"));
        set_input = reinterpret_cast<void (*)(const unsigned char *, std::size_t)>(::dlsym(handle, "bf_set_input"));
    }

    // Intentionally no dlclose: the process exits right after the run.
    ~LibraryHandle() = default;

    LibraryHandle(const LibraryHandle &) = delete;

    LibraryHandle &operator=(const LibraryHandle &) = delete;
};

static sigjmp_buf g_env;

static void on_fault(int) {
    siglongjmp(g_env, 1);
}

static bool call_guarded(const RunFn fn, unsigned char *base) {
    struct sigaction sa{};
    sa.sa_handler = on_fault;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_NODEFER;
    struct sigaction old_segv{};
    struct sigaction old_bus{};
    sigaction(SIGSEGV, &sa, &old_segv);
    sigaction(SIGBUS, &sa, &old_bus);

    volatile bool ok = true;
    if (sigsetjmp(g_env, 1) == 0) {
        fn(base);
    } else {
        ok = false;
    }

    sigaction(SIGSEGV, &old_segv, nullptr);
    sigaction(SIGBUS, &old_bus, nullptr);
    return ok;
}

struct TapeMap {
    void *region{nullptr};
    std::size_t total{0};
    unsigned char *base{nullptr};
};

static TapeMap alloc_tape() {
    const auto page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    const std::size_t tape_bytes = (TAPE_SIZE + page - 1) / page * page;
    const std::size_t guard = (GUARD_SIZE + page - 1) / page * page;
    TapeMap t;
    t.total = guard * 2 + tape_bytes;

    t.region = ::mmap(nullptr, t.total, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
    if (t.region == MAP_FAILED) throw std::runtime_error{"mmap failed"};

    auto *tape_start = static_cast<unsigned char *>(t.region) + guard;
    if (::mprotect(tape_start, tape_bytes, PROT_READ | PROT_WRITE) != 0) {
        ::munmap(t.region, t.total);
        throw std::runtime_error{"mprotect failed"};
    }
    t.base = tape_start + TAPE_OFFSET;
    return t;
}

// Program input preloaded from a regular-file stdin (then both tiers read it from memory, no stdio involved).
static std::string g_stdin_data;
static bool g_stdin_loaded = false;

static void preload_stdin_if_regular() {
    struct stat st{};
    if (::fstat(STDIN_FILENO, &st) != 0 || !S_ISREG(st.st_mode)) return;
    std::string data;
    char buf[65536];
    for (;;) {
        const ssize_t n = ::read(STDIN_FILENO, buf, sizeof buf);
        if (n <= 0) break;
        data.append(buf, static_cast<std::size_t>(n));
    }
    g_stdin_data = std::move(data);
    g_stdin_loaded = true;
}

static void run_native(const LibraryHandle &lib) {
    if (g_stdin_loaded && lib.set_input) {
        lib.set_input(reinterpret_cast<const unsigned char *>(g_stdin_data.data()), g_stdin_data.size());
    }
    const TapeMap tape = alloc_tape();
    if (!call_guarded(lib.fn, tape.base)) {
        if (lib.flush) lib.flush();
        ::munmap(tape.region, tape.total);
        throw std::runtime_error{"Pointer out of bounds"};
    }
    ::munmap(tape.region, tape.total);
}

// ---------------------------------------------------------------------------
// Tier 0: an in-process interpreter of the optimized IR. Used while the native build is not ready yet.
// ---------------------------------------------------------------------------
struct InterpState {
    const std::vector<Ir> *code{nullptr};
    const std::string *input{nullptr};
    std::size_t in_pos{0};
    std::string out;
    bool committed{false}; // output already left the process: the run can no longer be restarted natively
    bool aborted{false};

    bool (*poll)(void *){nullptr};

    void *poll_ctx{nullptr};
};

static InterpState *g_interp = nullptr;

static void interp_write(const std::string &data) {
    std::size_t off = 0;
    while (off < data.size()) {
        const ssize_t n = ::write(STDOUT_FILENO, data.data() + off, data.size() - off);
        if (n <= 0) break;
        off += static_cast<std::size_t>(n);
    }
}

static void interp_entry(unsigned char *base) {
    InterpState &s = *g_interp;
    const Ir *const code = s.code->data();
    unsigned char *p = base;
    std::size_t pc = 0;
    std::uint32_t budget = 1U << 20;

    // Called on every backward edge; every ~1M of them asks whether the native build became ready.
    const auto tick = [&s, &budget]() -> bool {
        if (--budget != 0) return false;
        budget = 1U << 20;
        if (!s.committed && s.poll != nullptr && s.poll(s.poll_ctx)) {
            s.aborted = true;
            return true;
        }
        return false;
    };

    for (;;) {
        const Ir &in = code[pc];
        switch (in.op) {
            case Op::Add:
                p[in.off] = static_cast<unsigned char>(p[in.off] + in.arg);
                ++pc;
                break;
            case Op::Set:
                p[in.off] = static_cast<unsigned char>(in.arg);
                ++pc;
                break;
            case Op::Move:
                p += in.arg;
                ++pc;
                break;
            case Op::Out:
                s.out.push_back(static_cast<char>(p[in.off]));
                ++pc;
                if (s.committed) {
                    if (s.out.size() >= 65536) {
                        interp_write(s.out);
                        s.out.clear();
                    }
                } else if (s.out.size() >= (4U << 20)) {
                    interp_write(s.out);
                    s.out.clear();
                    s.committed = true;
                }
                break;
            case Op::In:
                if (s.in_pos < s.input->size()) {
                    p[in.off] = static_cast<unsigned char>((*s.input)[s.in_pos++]);
                } else {
                    p[in.off] = 0;
                }
                ++pc;
                break;
            case Op::JumpIfZero:
            case Op::IfNonZero:
                pc = p[in.off] == 0 ? static_cast<std::size_t>(in.arg) + 1 : pc + 1;
                break;
            case Op::EndIf:
                ++pc;
                break;
            case Op::JumpIfNotZero:
                if (p[in.off] != 0) {
                    pc = static_cast<std::size_t>(in.arg) + 1;
                    if (tick()) return;
                } else {
                    ++pc;
                }
                break;
            case Op::MoveJnz:
                p += in.off2;
                if (p[in.off] != 0) {
                    pc = static_cast<std::size_t>(in.arg) + 1;
                    if (tick()) return;
                } else {
                    ++pc;
                }
                break;
            case Op::MulAdd:
                p[in.off2] = static_cast<unsigned char>(p[in.off2] + static_cast<unsigned>(in.arg) * p[in.off]);
                ++pc;
                break;
            case Op::MulAddClear:
                p[in.off2] = static_cast<unsigned char>(p[in.off2] + static_cast<unsigned>(in.arg) * p[in.off]);
                p[in.off] = 0;
                ++pc;
                break;
            case Op::MulCells:
                p[in.off2] = static_cast<unsigned char>(p[in.off2] +
                                                        static_cast<unsigned>(in.arg) * p[in.off] * p[in.off3]);
                ++pc;
                break;
            case Op::Scan:
                while (*p != 0) {
                    p += in.arg;
                    if (tick()) return;
                }
                ++pc;
                break;
            case Op::Halt:
                return;
        }
    }
}

// Returns true when the program ran to completion (its output is written), false when it was aborted so that the
// native build can run the program from scratch (nothing has been written in that case).
static bool run_interp(const ProgramIr &prog, const std::string &input, bool (*poll)(void *), void *poll_ctx) {
    InterpState state;
    state.code = &prog.code;
    state.input = &input;
    state.out = prog.pre_output;
    state.poll = poll;
    state.poll_ctx = poll_ctx;
    g_interp = &state;

    const TapeMap tape = alloc_tape();
    const bool ok = call_guarded(interp_entry, tape.base);
    ::munmap(tape.region, tape.total);
    if (!ok) {
        interp_write(state.out);
        throw std::runtime_error{"Pointer out of bounds"};
    }
    if (state.aborted) return false;
    interp_write(state.out);
    return true;
}

struct Race {
    pid_t pid{-1};
    bool ready{false};
    bool failed{false};
};

static bool race_poll(void *ctx) {
    auto &r = *static_cast<Race *>(ctx);
    if (r.ready || r.failed || r.pid <= 0) return r.ready;
    int st = 0;
    if (::waitpid(r.pid, &st, WNOHANG) == r.pid) {
        r.pid = -1;
        if (WIFEXITED(st) && WEXITSTATUS(st) == 0) r.ready = true;
        else r.failed = true;
    }
    return r.ready;
}

static pid_t spawn_compile(const BuildPlan &plan) {
    const std::string tmp = shell_quote(plan.tmp.string());
    const std::string cmd = "if " + plan.cmd + "; then mv -f " + tmp + " " + shell_quote(plan.lib) +
                            "; s=$?; else s=1; fi; rm -f " + shell_quote(plan.src.string()) + " " + tmp + "; exit $s";
    const pid_t pid = ::fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        ::setpgid(0, 0);
        const int dn = ::open("/dev/null", O_RDWR);
        if (dn >= 0) {
            ::dup2(dn, 0);
            ::dup2(dn, 1);
            ::dup2(dn, 2);
        }
        ::execl("/bin/sh", "sh", "-c", cmd.c_str(), static_cast<char *>(nullptr));
        ::_exit(127);
    }
    ::setpgid(pid, pid);
    return pid;
}

// Runs the program. Cache hit: native immediately. Cache miss: interpret right away while the native build runs in
// the background; whichever finishes first wins (a finished build restarts the program natively from scratch).
static void run_tiered(const std::string &program, const bool no_cache) {
    const bool has_input = program.find(',') != std::string::npos;
    if (has_input) preload_stdin_if_regular();

    const bool eligible = std::getenv("BF_NO_TIER") == nullptr && (!has_input || g_stdin_loaded);
    const BuildPlan probe = plan_build(program, no_cache, {}, {}, nullptr, false);
    if (probe.cached || !eligible) {
        const LibraryHandle lib{build_library(program, no_cache)};
        run_native(lib);
        return;
    }

    const ProgramIr prog = convert(program);
    const BuildPlan plan = plan_build(program, no_cache, {}, {}, &prog, true);
    Race race;
    race.pid = spawn_compile(plan);
    if (race.pid < 0) {
        const LibraryHandle lib{build_library(program, no_cache)};
        run_native(lib);
        return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    static const std::string no_input;
    const bool finished = run_interp(prog, g_stdin_loaded ? g_stdin_data : no_input, race_poll, &race);
    if (!finished) {
        const LibraryHandle lib{plan.lib};
        run_native(lib);
        return;
    }
    if (race.pid > 0) {
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (secs < 0.25) {
            // Short program: the build is not needed, drop it. A long-running one lets the build finish
            // (detached) so that the next run starts natively.
            ::kill(-race.pid, SIGKILL);
            ::waitpid(race.pid, nullptr, 0);
            std::error_code ec;
            fs::remove(plan.src, ec);
            fs::remove(plan.tmp, ec);
        }
    }
}

// llvm-profdata must match the compiler's raw profile version, so prefer the one sitting next to the compiler.
static std::string find_profdata(const std::string &cc) {
    if (const char *e = std::getenv("BF_PROFDATA")) return e;
    if (const fs::path real = resolve_cc(cc); !real.empty()) {
        if (const fs::path sib = real.parent_path() / "llvm-profdata"; ::access(sib.c_str(), X_OK) == 0) {
            return shell_quote(sib.string());
        }
    }
    return "xcrun llvm-profdata";
}

// Profile path depends on the program, toolchain and every codegen knob except `pgo` itself.
static fs::path pgo_profile_path(const std::string &program, const Toolchain &tc) {
    Config c = g_cfg;
    c.pgo = 0;
    const std::string base = std::format(
        "{:016x}", fnv1a(program + '\n' + tc.cc_id + '\n' + tc.flags + '\n' + config_tag(c) + "\npgo-v3"));
    return tc.dir / ("bf_" + base + ".profdata");
}

// Runs an instrumented build of the program and merges the raw profile. With `quiet` the run happens in a forked
// child with stdio on /dev/null (used by --tune); otherwise it is the user's real run in this process.
static void train_pgo(const std::string &program, const bool quiet) {
    const Toolchain tc = get_toolchain();
    const fs::path prof = pgo_profile_path(program, tc);
    const fs::path raw = tc.dir / std::format("{}.{}.profraw", prof.stem().string(), ::getpid());
    ::setenv("LLVM_PROFILE_FILE", raw.c_str(), 1);

    const Config saved = g_cfg;
    g_cfg.pgo = 0;
    const std::string gen = build_library(program, true, "-fprofile-instr-generate -DBF_PGO", "pgo-gen");
    g_cfg = saved;

    const auto run = [&gen] {
        const LibraryHandle lib{gen};
        run_native(lib);
        if (lib.dump) lib.dump();
    };
    if (quiet) {
        const pid_t pid = ::fork();
        if (pid < 0) throw std::runtime_error{"fork failed"};
        if (pid == 0) {
            const int dn = ::open("/dev/null", O_RDWR);
            if (dn >= 0) {
                ::dup2(dn, 0);
                ::dup2(dn, 1);
                ::dup2(dn, 2);
            }
            ::alarm(300);
            try {
                run();
            } catch (...) {
                ::_exit(1);
            }
            ::_exit(0);
        }
        int st = 0;
        ::waitpid(pid, &st, 0);
        if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) throw std::runtime_error{"PGO training run failed"};
    } else {
        run();
    }

    const fs::path tmp = tc.dir / std::format("{}.{}.tmp.profdata", prof.stem().string(), ::getpid());
    const std::string cmd = find_profdata(tc.cc) + " merge -output=" + shell_quote(tmp.string()) + " " +
                            shell_quote(raw.string());
    const int rc = std::system(cmd.c_str());
    std::error_code ec;
    fs::remove(raw, ec);
    if (rc != 0) {
        fs::remove(tmp, ec);
        throw std::runtime_error{"llvm-profdata merge failed (set BF_PROFDATA to its path)"};
    }
    fs::rename(tmp, prof);
}

// --pgo: the first run trains (it is the real run of the program), later runs use the profile.
static void run_pgo(const std::string &program, const bool no_cache) {
    const Toolchain tc = get_toolchain();
    g_cfg.pgo = 1;
    if (no_cache || !fs::exists(pgo_profile_path(program, tc))) {
        train_pgo(program, false);
        (void) build_library(program, true); // warm the cache for the next run
        return;
    }
    const LibraryHandle lib{build_library(program, no_cache)};
    run_native(lib);
}

// ---------------------------------------------------------------------------
// Autotuner: builds a handful of variants, times them on the real program (stdin/stdout -> /dev/null)
// and stores the fastest combination. Only for programs that don't read input.
// ---------------------------------------------------------------------------
static fs::path tune_file(const std::string &program, const Toolchain &tc) {
    const std::string key = std::format("{:016x}", fnv1a(program + '\n' + tc.cc_id + '\n' + tc.flags + "\ntune-v1"));
    return tc.dir / ("bf_" + key + ".tune");
}

static void load_tuned(const std::string &program) {
    const Toolchain tc = get_toolchain();
    std::ifstream f{tune_file(program, tc)};
    Config c;
    if (f >> c.scan >> c.flagset >> c.promote >> c.branchless >> c.unroll >> c.strided >> c.pgo && c.pgo >= 0 && c.pgo
        <= 1 && c.strided >= 0 && c.strided <= 1 && c.unroll >= 2 && c.unroll <= 16 &&
        c.branchless >= 0 && c.branchless <= 2 && c.scan >= 0 && c.scan <= 3 && c.flagset >= 0 && c.flagset <= 4 &&
        c.promote >= 4 && c.promote <= 128) {
        g_cfg = c;
    }
}

static double time_variant(const std::string &program, const Config &cfg, const int reps) {
    g_cfg = cfg;
    std::string path;
    try {
        path = build_library(program, false);
    } catch (const std::exception &) {
        return 1e30;
    }
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        const pid_t pid = ::fork();
        if (pid < 0) return 1e30;
        if (pid == 0) {
            const int dn = ::open("/dev/null", O_RDWR);
            if (dn >= 0) {
                ::dup2(dn, 0);
                ::dup2(dn, 1);
                ::dup2(dn, 2);
            }
            ::alarm(300);
            try {
                const LibraryHandle lib{path};
                run_native(lib);
            } catch (...) {
                ::_exit(1);
            }
            ::_exit(0);
        }
        int st = 0;
        ::waitpid(pid, &st, 0);
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) return 1e30;
        best = std::min(best, dt);
    }
    return best;
}

static void run_tune(const std::string &program) {
    if (program.find(',') != std::string::npos) {
        throw std::runtime_error{"--tune needs a program without input (',')"};
    }
    Config best;
    double best_t = time_variant(program, best, 3);
    if (best_t > 1e29) throw std::runtime_error{"baseline variant failed to build or run"};
    std::println(stderr, "baseline {:<12} {:.3f}s", config_tag(best), best_t);

    const auto try_cfg = [&](const Config &cand) {
        const double t = time_variant(program, cand, 3);
        std::println(stderr, "         {:<12} {:.3f}s", config_tag(cand), t);
        if (t < best_t * 0.98) {
            best_t = t;
            best = cand;
        }
    };
    for (const int v: {1, 0}) {
        Config cand = best;
        cand.scan = v;
        try_cfg(cand);
    }
    {
        Config cand = best;
        cand.strided = 0;
        try_cfg(cand);
    }
    for (const int v: {2, 6, 8}) {
        Config cand = best;
        cand.unroll = v;
        try_cfg(cand);
    }
    for (int v = 1; v <= 4; ++v) {
        Config cand = best;
        cand.flagset = v;
        try_cfg(cand);
    }
    for (const int v: {1, 2}) {
        Config cand = best;
        cand.branchless = v;
        try_cfg(cand);
    }
    for (const int v: {12, 20, 40, 64}) {
        Config cand = best;
        cand.promote = v;
        try_cfg(cand);
    }

    try {
        g_cfg = best;
        train_pgo(program, true);
        Config cand = best;
        cand.pgo = 1;
        try_cfg(cand);
    } catch (const std::exception &e) {
        std::println(stderr, "         pgo skipped: {}", e.what());
    }

    const Toolchain tc = get_toolchain();
    std::ofstream out{tune_file(program, tc), std::ios::trunc};
    out << best.scan << ' ' << best.flagset << ' ' << best.promote << ' ' << best.branchless << ' ' << best.unroll <<
            ' ' << best.strided << ' ' << best.pgo << '\n';
    g_cfg = best;
    std::println(stderr, "selected {} ({:.3f}s)", config_tag(best), best_t);
}

static std::string read_file(const std::string &path) {
    std::ifstream file{path, std::ios::binary | std::ios::ate};
    if (!file) throw std::runtime_error{"Cannot open file: " + path};
    const std::streamsize size = file.tellg();
    if (size < 0) throw std::runtime_error{"Failed to read file size: " + path};
    std::string buffer(static_cast<std::size_t>(size), '\0');
    file.seekg(0);
    file.read(buffer.data(), size);
    return buffer;
}

int main(const int argc, char **argv) {
    bool emit_c = false;
    bool no_cache = false;
    bool pgo = false;
    bool tune = false;
    std::string path;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "--emit-c") emit_c = true;
        else if (arg == "--no-cache") no_cache = true;
        else if (arg == "--pgo") pgo = true;
        else if (arg == "--tune") tune = true;
        else path = argv[i];
    }

    try {
        if (path.empty()) {
            std::println(stderr, "Usage: {} [--emit-c] [--no-cache] [--pgo] [--tune] <program.bf>", argv[0]);
            return 2;
        }
        const std::string program = read_file(path);

        if (tune) {
            run_tune(program);
            return 0;
        }
        load_tuned(program);
        if (g_cfg.pgo && !fs::exists(pgo_profile_path(program, get_toolchain()))) g_cfg.pgo = 0;
        if (emit_c) {
            const std::string c_source = generate_c(convert(program));
            std::fputs(c_source.c_str(), stdout);
            return 0;
        }
        if (pgo) {
            run_pgo(program, no_cache);
            std::fflush(nullptr);
            ::_exit(0);
        }
        run_tiered(program, no_cache);
        std::fflush(nullptr);
        ::_exit(0);
    } catch (const std::exception &e) {
        std::println(stderr, "Error: {}", e.what());
        return 1;
    }
    return 0;
}
