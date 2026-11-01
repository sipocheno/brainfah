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

static std::vector<std::size_t> preprocessing(const std::string &program) {
    const std::size_t length{program.length()};
    std::vector<std::size_t> stack{};
    std::vector<std::size_t> jumps(length);
    for (std::size_t i{0}; i < length; ++i) {
        if (program[i] == '[') {
            stack.push_back(i);
        } else if (program[i] == ']') {
            if (stack.empty()) {
                throw std::runtime_error{"Extra ] at position " + std::to_string(i)};
            }
            const std::size_t j = stack.back();
            stack.pop_back();
            jumps[i] = j;
            jumps[j] = i;
        }
    }
    if (!stack.empty()) {
        throw std::runtime_error{"Unclosed [ at position " + std::to_string(stack.back())};
    }
    return jumps;
}

static void run(const std::string &program, const std::vector<std::size_t> &jump_table) {
    std::array<unsigned char, TAPE_SIZE> bytes{};
    std::size_t ptr{};
    std::size_t pc{};
    while (pc < program.length()) {
        switch (program[pc]) {
            case '>':
                ptr = (ptr + 1 == TAPE_SIZE) ? 0 : ptr + 1;
                break;
            case '<':
                ptr = (ptr == 0) ? TAPE_SIZE - 1 : ptr - 1;
                break;
            case '+':
                ++bytes[ptr];
                break;
            case '-':
                --bytes[ptr];
                break;
            case '.':
                std::putchar(bytes[ptr]);
                break;
            case ',': {
                std::fflush(stdout);
                const int c = std::cin.get();
                bytes[ptr] = (c == EOF) ? 0 : static_cast<unsigned char>(c);
                break;
            }
            case '[':
                if (bytes[ptr] == 0) pc = jump_table[pc];
                break;
            case ']':
                if (bytes[ptr] != 0) pc = jump_table[pc];
                break;
            default:
                break;
        }
        ++pc;
    }
    std::fflush(stdout);
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
        const auto jump_table = preprocessing(program);
        run(program, jump_table);
    } catch (const std::exception &e) {
        std::println(stderr, "Error: {}", e.what());
        return 1;
    }
    return 0;
}
