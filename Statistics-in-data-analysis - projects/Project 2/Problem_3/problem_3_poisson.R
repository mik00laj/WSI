# Instalacja i załadowanie pakietu ggplot2
install.packages("ggplot2")
library(ggplot2)

# Ustawienie generatora liczb losowych
set.seed(123)

# Lista różnych liczności prób
n_values <- c(10, 100, 1000, 10000, 100000, 1000000)

# Liczba prób
m <- 5000

# Parametry
lambda <- 0.5
mu <- 0
sigma <- 1

# Tworzenie pustej ramki danych do przechowywania wyników
df <- data.frame()

# Generowanie próbek i obliczanie estymatorów największej wiarygodności
for (n in n_values) {
  estymatory <- numeric(m)
  for (i in 1:m) {
    probka <- rpois(n, lambda = lambda)
    estymatory[i] <- mean(probka)
  }
  I <- 1 / lambda
  estymatory <- (estymatory - lambda) * sqrt(I * n)
  # Test normalności estymatorów
  print(shapiro.test(estymatory))
  # Dodawanie wyników do ramki danych
  df <- rbind(df, data.frame(estymatory = estymatory, n = as.factor(n)))
}

# Wykres histogramów estymatorów z dodaniem rozkładu normalnego
ggplot(df, aes(x = estymatory, fill = n)) +
  geom_histogram(aes(y = ..density..), bins = 30, color = "black", alpha = 0.6) +
  stat_function(fun = dnorm, args = list(mean = mu, sd = sigma), color = "red", lwd = 2, linetype = "dashed") +
  labs(
    title = "Rozkład estymatorów największej wiarygodności",
    x = expression(sqrt(n * I(lambda)) * (hat(lambda) - lambda)),
    y = "Gęstość"
  ) +
  theme_minimal() +
  facet_wrap(~ n, scales = "free_y", nrow = 2)
