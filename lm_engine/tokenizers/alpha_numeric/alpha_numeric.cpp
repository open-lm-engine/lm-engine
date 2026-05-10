// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

struct TokenizerOutput {
    std::vector<std::vector<int>> input_ids;
    std::vector<std::vector<int>> attention_mask;
};

class AlphaNumericTokenizer {
public:
    static constexpr int eos_token_id = 62;
    static constexpr int pad_token_id = eos_token_id;
    static constexpr const char *eos_token_str = "<|endoftext|>";

    AlphaNumericTokenizer() = default;

    TokenizerOutput encode_batch(const std::vector<std::string> &inputs,
                                 bool padding = false,
                                 bool add_special_tokens = true) const {
        std::vector<std::vector<int>> ids;
        ids.reserve(inputs.size());

        for (const auto &sample : inputs) {
            std::vector<int> seq;
            seq.reserve(sample.size() + (add_special_tokens ? 1 : 0));
            for (char c : sample)
                seq.push_back(get_token_id(c));
            if (add_special_tokens)
                seq.push_back(eos_token_id);
            ids.push_back(std::move(seq));
        }

        if (!padding && ids.size() > 1) {
            size_t len = ids[0].size();
            for (const auto &seq : ids)
                if (seq.size() != len)
                    throw std::invalid_argument("padding must be true for sequences of unequal length");
        }

        TokenizerOutput out;
        if (padding) {
            size_t max_len = 0;
            for (const auto &seq : ids)
                max_len = std::max(max_len, seq.size());

            out.input_ids.reserve(ids.size());
            out.attention_mask.reserve(ids.size());

            for (const auto &seq : ids) {
                size_t pad_len = max_len - seq.size();

                std::vector<int> padded(pad_len, pad_token_id);
                padded.insert(padded.end(), seq.begin(), seq.end());

                std::vector<int> mask(pad_len, 0);
                mask.insert(mask.end(), seq.size(), 1);

                out.input_ids.push_back(std::move(padded));
                out.attention_mask.push_back(std::move(mask));
            }
        } else {
            out.input_ids = std::move(ids);
        }

        return out;
    }

    std::vector<int> encode(const std::string &input, bool add_special_tokens = true) const {
        return encode_batch({input}, false, add_special_tokens).input_ids[0];
    }

    int get_token_id(char c) const {
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc >= '0' && uc <= '9')
            return uc - '0';
        if (uc >= 'a' && uc <= 'z')
            return uc - 'a' + 10;
        if (uc >= 'A' && uc <= 'Z')
            return uc - 'A' + 36;
        throw std::invalid_argument(std::string("unexpected token: ") + c);
    }

    std::string decode(const std::vector<int> &ids, bool skip_special_tokens = true) const {
        std::string out;
        out.reserve(ids.size());
        for (int id : ids) {
            if (id == eos_token_id) {
                if (!skip_special_tokens)
                    out += eos_token_str;
                continue;
            }
            out += get_char(id);
        }
        return out;
    }

    std::vector<std::string> batch_decode(const std::vector<std::vector<int>> &ids,
                                          bool skip_special_tokens = true) const {
        std::vector<std::string> out;
        out.reserve(ids.size());
        for (const auto &seq : ids)
            out.push_back(decode(seq, skip_special_tokens));
        return out;
    }

    char get_char(int id) const {
        if (id >= 0 && id <= 9)
            return static_cast<char>('0' + id);
        if (id >= 10 && id <= 35)
            return static_cast<char>('a' + id - 10);
        if (id >= 36 && id <= 61)
            return static_cast<char>('A' + id - 36);
        throw std::invalid_argument("unexpected token id: " + std::to_string(id));
    }
};

PYBIND11_MODULE(alpha_numeric_cpp, m) {
    py::class_<TokenizerOutput>(m, "TokenizerOutput")
        .def_readwrite("input_ids", &TokenizerOutput::input_ids)
        .def_readwrite("attention_mask", &TokenizerOutput::attention_mask);

    py::class_<AlphaNumericTokenizer>(m, "AlphaNumericTokenizer")
        .def(py::init<>())
        .def("encode_batch",
             &AlphaNumericTokenizer::encode_batch,
             py::arg("inputs"),
             py::arg("padding") = false,
             py::arg("add_special_tokens") = true)
        .def("encode", &AlphaNumericTokenizer::encode, py::arg("input"), py::arg("add_special_tokens") = true)
        .def("decode", &AlphaNumericTokenizer::decode, py::arg("ids"), py::arg("skip_special_tokens") = true)
        .def("batch_decode",
             &AlphaNumericTokenizer::batch_decode,
             py::arg("ids"),
             py::arg("skip_special_tokens") = true);
}
