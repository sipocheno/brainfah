// Run:    ./brainfah tests/mandelbrot.bf
//         ./brainfah --emit-c tests/mandelbrot.bf
//         ./brainfah --no-cache tests/mandelbrot.bf

#include <algorithm>
#include <csignal>
#include <cstddef>
#include <cstdio>
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
        Scan, // while (*p) p += arg
        MoveJnz, // p += off2; }
        Halt
    };

    struct Ir {
        Op op;
        int arg{};
        int off{};
        int off2{};
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
        int x = a;
        x *= 2 - a * x;
        x *= 2 - a * x;
        x *= 2 - a * x;
        return x & 0xFF;
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
            auto &[op, arg, off, off2] = code[k];
            if (op == Op::MulAdd || op == Op::MulAddClear) {
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
                continue;
            }
            if (inner_depth == 0) {
                if ((ins.op == Op::Set && ins.off == base && mod256(ins.arg) == 0) ||
                    (ins.op == Op::MulAddClear && ins.off == base && ins.off2 != base)) {
                    base_zeroed = true;
                } else if (ins.off == base ||
                           ((ins.op == Op::MulAdd || ins.op == Op::MulAddClear) && ins.off2 == base)) {
                    base_zeroed = false;
                }
            } else {
                if (ins.off == base || ((ins.op == Op::MulAdd || ins.op == Op::MulAddClear) && ins.off2 == base)) {
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

        return false;
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
                const auto &[op, arg, off, off2] = ir[cur];
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
    const auto ensure_enclosing_loops_rebased = [&] {
        for (std::size_t s = 0; s < stack.size(); ++s) {
            if (stack[s].base != 0) {
                const int b = stack[s].base;
                rebase_loop_to_zero(result, stack[s]);
                // Adjust subsequent nested loops on the stack that were inside `stack[s]`
                for (std::size_t inner = s + 1; inner < stack.size(); ++inner) {
                    stack[inner].base -= b;
                    // Re-sync `open` index of inner loop if `rebase_loop_to_zero` inserted/erased a Move
                }
                pending -= b;
            }
        }
    };

    const auto add = [&](const int delta) {
        for (std::size_t i = result.size(); i-- > 0;) {
            auto &[op, arg, off, off2] = result[i];
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

    (void) ensure_enclosing_loops_rebased;
    return pre_evaluate(result);
}

// ---------------------------------------------------------------------------
// Backend: Multi-level SSA Register Promotion + Apple Silicon NEON C Codegen
// ---------------------------------------------------------------------------
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
            "typedef uint8_t u8;\n\n"
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
            "#else\n"
            "static inline u8* scan_right_1(u8 *p) { return (u8 *)memchr(p, 0, (size_t)-1); }\n"
            "static inline u8* scan_left_1(u8 *p) { while (*p) --p; return p; }\n"
            "#endif\n\n"
            "__attribute__((noinline)) static size_t bf_flush(u8 *out_buf, size_t pos) {\n"
            "    if (pos > 0) (void)write(STDOUT_FILENO, out_buf, pos);\n"
            "    return 0;\n"
            "}\n\n"
            "static inline __attribute__((always_inline)) void bf_out(u8 c, u8 *out_buf, size_t *out_pos) {\n"
            "    out_buf[(*out_pos)++] = c;\n"
            "    if (__builtin_expect(*out_pos == 65536 || c == 10, 0)) {\n"
            "        *out_pos = bf_flush(out_buf, *out_pos);\n"
            "    }\n"
            "}\n\n"
            "__attribute__((noinline)) static u8 bf_get(u8 *out_buf, size_t *pos) {\n"
            "    *pos = bf_flush(out_buf, *pos);\n"
            "    int c = getchar_unlocked();\n"
            "    return (u8)(c == EOF ? 0 : c);\n"
            "}\n\n"
            "void bf_run(u8 * __restrict base) {\n"
            "    u8 * __restrict p = base;\n"
            "    u8 out_buf[65536];\n"
            "    size_t out_pos = 0;\n";

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
                line(std::format("if (p[{}]) {{", in.off));
                ++depth;
                break;
            case Op::EndIf:
                --depth;
                line("}");
                break;
            case Op::JumpIfZero: {
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

                if (can_promote && all_cells.size() <= 28) {
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
                            case Op::JumpIfZero:
                                line(std::format("while ({}) {{", make_reg(body.off)));
                                ++depth;
                                break;
                            case Op::JumpIfNotZero:
                                --depth;
                                line("}");
                                break;
                            case Op::IfNonZero:
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
            case Op::Scan:
                if (in.arg == 1) {
                    line("p = scan_right_1(p);");
                } else if (in.arg == -1) {
                    line("p = scan_left_1(p);");
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

static std::string build_library(const std::string &c_source, const bool no_cache) {
    namespace fs = std::filesystem;
#ifdef __APPLE__
    const char *ext = ".dylib";
    const char *default_flags =
            "-O3 -mcpu=native -fomit-frame-pointer -fno-stack-protector -fstrict-aliasing -fvectorize -fslp-vectorize";
#else
    const char *ext = ".so";
    const char *default_flags = "-O3 -march=native -fomit-frame-pointer -fno-stack-protector -fstrict-aliasing";
#endif
    const char *cc_env = std::getenv("BF_CC");
    const char *flags_env = std::getenv("BF_CFLAGS");
    const char *dir_env = std::getenv("BF_CACHE_DIR");

    const std::string cc = cc_env ? cc_env : "clang";
    const std::string flags = std::string{flags_env ? flags_env : default_flags} + " -fno-finite-loops";

    const fs::path dir = dir_env ? fs::path{dir_env} : fs::temp_directory_path() / "bf_native_cache";
    fs::create_directories(dir);

    const std::string key = std::format("{:016x}", fnv1a(c_source + '\n' + cc + '\n' + flags));
    const fs::path lib = dir / ("bf_" + key + ext);

    if (no_cache || !fs::exists(lib)) {
        const fs::path tmp = dir / std::format("bf_{}_{}.tmp{}", key, ::getpid(), ext);
        const std::string cmd = cc + " " + flags + " -fPIC -shared -w -x c - -o " + shell_quote(tmp.string());

        FILE *pipe = ::popen(cmd.c_str(), "w");
        if (!pipe) throw std::runtime_error{"popen failed"};
        const std::size_t written = std::fwrite(c_source.data(), 1, c_source.size(), pipe);
        const int rc = ::pclose(pipe);

        if (written != c_source.size() || rc != 0) {
            std::error_code ec;
            fs::remove(tmp, ec);
            throw std::runtime_error{"Compilation failed (command: " + cmd + ")"};
        }
        fs::rename(tmp, lib);
    }
    return lib.string();
}

using RunFn = void (*)(unsigned char *);

struct LibraryHandle {
    void *handle{nullptr};
    RunFn fn{nullptr};

    explicit LibraryHandle(const std::string &path) {
        handle = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) throw std::runtime_error{std::string{"dlopen failed: "} + ::dlerror()};
        void *sym = ::dlsym(handle, "bf_run");
        if (!sym) {
            ::dlclose(handle);
            throw std::runtime_error{"bf_run symbol not found"};
        }
        fn = reinterpret_cast<RunFn>(sym);
    }

    ~LibraryHandle() {
        if (handle) ::dlclose(handle);
    }

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

static void run_native(const RunFn fn) {
    const auto page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    const std::size_t tape_bytes = (TAPE_SIZE + page - 1) / page * page;
    const std::size_t guard = (GUARD_SIZE + page - 1) / page * page;
    const std::size_t total = guard * 2 + tape_bytes;

    void *region = ::mmap(nullptr, total, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (region == MAP_FAILED) throw std::runtime_error{"mmap failed"};

    auto *tape_start = static_cast<unsigned char *>(region) + guard;
    if (::mprotect(tape_start, tape_bytes, PROT_READ | PROT_WRITE) != 0) {
        ::munmap(region, total);
        throw std::runtime_error{"mprotect failed"};
    }

    unsigned char *base = tape_start + TAPE_OFFSET;
    const bool ok = call_guarded(fn, base);
    ::munmap(region, total);
    if (!ok) throw std::runtime_error{"Pointer out of bounds"};
}

static bool use_sample() {
    std::print("Do you want to use sample hello world program for test? (y/n)\n>");
    std::fflush(stdout);
    std::string answer;
    std::getline(std::cin, answer);
    return !answer.empty() && (answer[0] == 'y' || answer[0] == 'Y');
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
    std::string path;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "--emit-c") emit_c = true;
        else if (arg == "--no-cache") no_cache = true;
        else path = argv[i];
    }

    try {
        std::string program{};
        if (!path.empty()) {
            program = read_file(path);
        } else {
            program =
                    "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>\n"
                    "---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.";
            if (!use_sample()) {
                std::println("Enter your brainfuck program:");
                std::fflush(stdout);
                std::getline(std::cin, program);
            }
        }

        const std::string c_source = generate_c(convert(program));
        if (emit_c) {
            std::fputs(c_source.c_str(), stdout);
            return 0;
        }
        const LibraryHandle lib{build_library(c_source, no_cache)};
        run_native(lib.fn);
    } catch (const std::exception &e) {
        std::println(stderr, "Error: {}", e.what());
        return 1;
    }
    return 0;
}
