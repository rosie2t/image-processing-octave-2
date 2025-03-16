function [encoded, dict] = huffman_encode(Im)
  symbols = unique(Im(:));
  freq = histc(Im(:), symbols);
  dict = huffmandict(symbols, freq);
  encoded = huffmanenco(Im(:), dict);
end