function decoded = huffman_decode(encoded, dict, Im_size)
  decoded = huffmandeco(encoded, dict);
  decoded = reshape(decoded, Im_size);
end