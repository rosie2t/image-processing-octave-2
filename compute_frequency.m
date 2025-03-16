function freq = compute_frequency(Im)
  freq = zeros(256, 1);%Δεδομένου ότι έχουμε 256 πιθανές τιμές pixel (0-255)
  [rows, cols] = size(Im);
  for i = 1:rows
    for j = 1:cols
      pixel_value = Im(i, j);
      freq(pixel_value + 1) += 1;
    end
  end
end
