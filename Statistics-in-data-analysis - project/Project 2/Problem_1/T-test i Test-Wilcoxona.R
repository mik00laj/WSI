# Instalacja i załadowanie pakietu ggplot2
install.packages("ggplot2")
library(ggplot2)

# Ustawienie generatora liczb losowych
set.seed(123)

# Funkcja do symulacji i obliczania mocy testu t-Studenta
calculate_power_t_test <- function(n, delta, sigma, alpha, num_simulations) {
  power <- numeric(length(delta))
  
  for (i in 1:length(delta)) {
    reject_count <- 0
    
    for (j in 1:num_simulations) {
      x <- rnorm(n, mean = 0, sd = sigma)
      y <- rnorm(n, mean = delta[i], sd = sigma)
      
      p_value <- t.test(x, y)$p.value
      
      if (p_value < alpha) {
        reject_count <- reject_count + 1
      }
    }
    
    power[i] <- reject_count / num_simulations
  }
  
  return(power)
}

# Funkcja do symulacji i obliczania mocy testu Wilcoxona
calculate_power_wilcoxon_test <- function(n, delta, sigma, alpha, num_simulations) {
  power <- numeric(length(delta))
  
  for (i in 1:length(delta)) {
    reject_count <- 0
    
    for (j in 1:num_simulations) {
      x <- rnorm(n, mean = 0, sd = sigma)
      y <- rnorm(n, mean = delta[i], sd = sigma)
      
      p_value <- wilcox.test(x, y)$p.value
      
      if (p_value < alpha) {
        reject_count <- reject_count + 1
      }
    }
    
    power[i] <- reject_count / num_simulations
  }
  
  return(power)
}

# Parametry
n <- 30
sigma <- 1
alpha <- 0.05
num_simulations <- 1000
delta <- seq(-2, 2, by = 0.1)

# Obliczenie mocy testu t-Studenta
power_t_test <- calculate_power_t_test(n, delta, sigma, alpha, num_simulations)

# Obliczenie mocy testu Wilcoxona
power_wilcoxon_test <- calculate_power_wilcoxon_test(n, delta, sigma, alpha, num_simulations)

# Przygotowanie danych do wykresu
df <- data.frame(
  delta = rep(delta, 2),
  power = c(power_t_test, power_wilcoxon_test),
  test = rep(c("t-Studenta", "Wilcoxona"), each = length(delta))
)

# Wykres mocy testów
ggplot(df, aes(x = delta, y = power, color = test)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  geom_hline(yintercept = 0.8, linetype = "dashed", color = "red") +
  labs(title = "Moc testów t-Studenta i Wilcoxona",
       x = expression(Delta),
       y = "Moc",
       color = "Test") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_color_manual(values = c("blue", "green"))
