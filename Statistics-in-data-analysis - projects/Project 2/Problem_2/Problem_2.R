# Wczytanie danych
inflacja_oficjalna <- c(0, 0.7, 0.1, 2.5, 1.2, 1.1, 0.7, 0, 0, -0.2, 0, -0.4, 0.3, 0.7, 0.1, 0.4, 0.3, 0.2)
inflacja_studencka <- c(0, 1.23, 0.59, 2.83, 1.9, 1.34, 1.01, 0.84, 0.93, 0.29, 0.57, 0.12, 0.73, 0.87, -0.28, 2.4, 0.64, 0.37)

# Krok 2: Rysowanie wykresów pudełkowych
boxplot(inflacja_oficjalna, inflacja_studencka,
        names = c("Inflacja Oficjalna", "Inflacja Studencka"),
        main = "Wykresy pudełkowe inflacji",
        ylab = "Wartość [%]")

# Krok 3: Wykres pudełkowy różnic
roznice <- inflacja_studencka - inflacja_oficjalna
boxplot(roznice, main = "Wykres pudełkowy różnic (Inflacja Studencka - Inflacja Oficjalna)",
        ylab = "Różnica [%]")

# Krok 4: Formułowanie hipotez
# H0: Mediany inflacji są równe
# H1: Mediany inflacji są różne

# Krok 5: Test normalności
test_normalnosci <- shapiro.test(roznice)

# Krok 6: Wybór testu i przeprowadzenie testu statystycznego

if (test_normalnosci$p.value > 0.05) {
  # Jeśli różnice są normalnie rozłożone, używamy t-testu dla prób sparowanych
  wynik_testu <- t.test(inflacja_studencka, inflacja_oficjalna, paired = TRUE)
} else {
  # Jeśli różnice nie są normalnie rozłożone, używamy testu Wilcoxona
  wynik_testu <- wilcox.test(inflacja_studencka, inflacja_oficjalna, paired = TRUE, exact = FALSE)
}
# Wyświetlenie wyników testu
print(wynik_testu)

# Krok 7: Interpretacja wyników
if (wynik_testu$p.value < 0.05) {
  cat("Odrzucamy hipotezę zerową. Istnieje statystycznie istotna różnica między inflacją studencką a oficjalną.\n")
} else {
  cat("Nie ma podstaw do odrzucenia hipotezy zerowej. Brak statystycznie istotnej różnicy między inflacją studencką a oficjalną.\n")
}
