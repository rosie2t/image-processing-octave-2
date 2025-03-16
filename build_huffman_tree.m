function huffman_tree = build_huffman_tree(freq)
  symbols = find(freq); % Βρίσκει τις μηδενικές συχνότητες
  prob = freq(symbols) / sum(freq); %Κανονικοποιεί τις συχνότητες σε πιθανότητες
  huffman_tree = huffmandict(symbols - 1, prob); % Δημιουργεί το λεξικό Huffman
end

