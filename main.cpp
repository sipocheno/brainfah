#include <array>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <iostream>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
constexpr std::size_t TAPE_SIZE{30000};

namespace {
    enum class Op { Add, Move, Out, In, JumpIfZero, JumpIfNotZero, Clear };
}

namespace {
    struct Instr {
        Op op;
        int arg;
    };
}

static bool use_sample() {
    std::print("Do you want to use sample hello world program for test? (y/n)\n>");
    std::fflush(stdout);
    std::string answer;
    std::getline(std::cin, answer);
    return !answer.empty() && (answer[0] == 'y' || answer[0] == 'Y');
}

static std::string read_file(const std::string &path) {
    std::ifstream file{path};
    if (!file) {
        throw std::runtime_error{"Cannot open file: " + path};
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

static void run(const std::vector<Instr> &commands) {
    std::array<unsigned char, TAPE_SIZE> bytes{};
    std::size_t pc{};
    std::size_t ptr{};
    while (pc < commands.size()) {
        switch (const auto [op, arg] = commands[pc]; op) {
            case Op::Add:
                bytes[ptr] = static_cast<unsigned char>(bytes[ptr] + arg);
                break;
            case Op::Move: {
                ptr = static_cast<std::size_t>(static_cast<std::ptrdiff_t>(ptr) + arg);
                if (ptr >= TAPE_SIZE) {
                    throw std::runtime_error{"Pointer out of bounds"};
                }
                break;
            }
            case Op::Out:
                std::putchar(bytes[ptr]);
                break;
            case Op::In: {
                std::fflush(stdout);
                const int c = std::cin.get();
                bytes[ptr] = (c == EOF) ? 0 : static_cast<unsigned char>(c);
                break;
            }
            case Op::JumpIfZero:
                if (bytes[ptr] == 0) pc = static_cast<std::size_t>(arg);
                break;
            case Op::JumpIfNotZero:
                if (bytes[ptr] != 0) pc = static_cast<std::size_t>(arg);
                break;
            case Op::Clear:
                bytes[ptr] = 0;
                break;
        }
        ++pc;
    }
    std::fflush(stdout);
}

static std::vector<Instr> convert(const std::string_view program) {
    std::vector<Instr> result{};
    std::vector<std::size_t> stack{};
    result.reserve(program.size());
    for (const char c: program) {
        if (c == '+' || c == '-') {
            const int d = (c == '+') ? 1 : -1;
            if (!result.empty() && result.back().op == Op::Add) {
                result.back().arg += d;
            } else {
                result.push_back({.op = Op::Add, .arg = d});
            }
        } else if (c == '>' || c == '<') {
            const int d = (c == '>') ? 1 : -1;
            if (!result.empty() && result.back().op == Op::Move) {
                result.back().arg += d;
            } else {
                result.push_back({.op = Op::Move, .arg = d});
            }
        } else if (c == '.') {
            result.push_back({.op = Op::Out, .arg = 0});
        } else if (c == ',') {
            result.push_back({.op = Op::In, .arg = 0});
        } else if (c == '[') {
            stack.push_back(result.size());
            result.push_back({.op = Op::JumpIfZero, .arg = 0});
        } else if (c == ']') {
            if (stack.empty()) {
                throw std::runtime_error{"Extra ]"};
            }
            if (const std::size_t size = result.size(); size >= 2
                                                        && result[size - 2].op == Op::JumpIfZero
                                                        && result[size - 1].op == Op::Add
                                                        && (result[size - 1].arg == 1 || result[size - 1].arg == -1)) {
                result.pop_back();
                result.pop_back();
                stack.pop_back();
                result.push_back({.op = Op::Clear, .arg = 0});
                continue;
            }
            const std::size_t j = stack.back();
            stack.pop_back();
            result.push_back({.op = Op::JumpIfNotZero, .arg = static_cast<int>(j)});
            result[j].arg = static_cast<int>(result.size() - 1);
        }
    }
    if (!stack.empty()) {
        throw std::runtime_error{"Unclosed ["};
    }
    return result;
}

int main(const int argc, char **argv) {
    std::string program{};
    try {
        if (argc >= 2) {
            program = read_file(argv[1]);
        } else {
            program =
                    "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>\n---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.";
            if (!use_sample()) {
                std::println("Enter your brainfuck program:");
                std::fflush(stdout);
                std::getline(std::cin, program);
            }
        }
        const auto vec = convert(program);
        run(vec);
    } catch (const std::exception &e) {
        std::println(stderr, "Error: {}", e.what());
        return 1;
    }
    return 0;
}
