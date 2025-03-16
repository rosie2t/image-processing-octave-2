function encoded_img = huffman_encode_image(Im, huffman_tree)
  [rows, cols] = size(Im);
  encoded_img = cell(rows, cols);
  for i = 1:rows
    for j = 1:cols
      encoded_img{i, j} = huffmanenco(Im(i, j), huffman_tree);
    end
  end
end
