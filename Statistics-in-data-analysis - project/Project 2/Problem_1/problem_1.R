install.packages("ggplot2")
library(ggplot2)  
set.seed(123)

# Funkcja obliczająca moc testu t-Studenta wykorzystująca symulacje komputerowe
simulate_power_t_test <- function(n, delta, mu_x, sigma_x, sigma_y, alpha, num_simulations) {
   reject_null <- 0

   for (i in 1:num_simulations) {
     x <- rnorm(n, mu_x, sigma_x)
     y <- rnorm(n, mu_x + delta, sigma_y)

     t_test_result <- t.test(x, y, alternative="two.sided", conf.level = 1 - alpha,
                             var.equal = FALSE)

     if (t_test_result$p.value < alpha) {
       reject_null <- reject_null + 1
     }
   }

   power <- reject_null / num_simulations

   return(power)
}

# Funkcja obliczająca moc testu sumy rank Wilcoxona wykorzystująca symulacje komputerowe
simulate_power_wilcoxon_test <- function(n, delta, mu_x, sigma_x, sigma_y, alpha, num_simulations) {
  reject_null <- 0

  for (i in 1:num_simulations) {
    x <- rnorm(n, mu_x, sigma_x)
    y <- rnorm(n, mu_x + delta, sigma_y)

    wilcox_test_result <- wilcox.test(x, y, alternative="two.sided", conf.level = 1 - alpha,
                                      var.equal = FALSE)

    if (wilcox_test_result$p.value < alpha) {
      reject_null <- reject_null + 1
    }
  }

  power <- reject_null / num_simulations

  return(power)
}

# Wywołanie obu funkcji dla przykładowych wartości parametrów i wektora delt
n <- 15
deltas <- seq(-2, 2, by=0.02)
alpha <- 0.05
num_simulations <- 1000
mu_x <- 0
sigma_x <- 1
sigma_y <- 2

powers_t <- sapply(deltas, function(delta)
  simulate_power_t_test(n, delta, mu_x, sigma_x, sigma_y, alpha=alpha, num_simulations=num_simulations))
powers_wilcoxon <- sapply(deltas, function(delta)
  simulate_power_wilcoxon_test(n, delta, mu_x, sigma_x, sigma_y, alpha=alpha, num_simulations=num_simulations))

# Przygotowanie danych do wykresu
df <- data.frame(
  delta = rep(deltas, 2),
  power = c(powers_t, powers_wilcoxon),
  test = rep(c("t-Studenta", "Wilcoxona"), each = length(deltas))
)

# Wykryte różnice na poziomie mocy 0.8
delta_t_0.8 <- deltas[which.min(abs(powers_t - 0.8))]
delta_wilcoxon_0.8 <- deltas[which.min(abs(powers_wilcoxon - 0.8))]

cat("Delta dla której moc testu t-Studenta osiąga 0.8:", delta_t_0.8, "\n")
cat("Delta dla której moc testu Wilcoxona osiąga 0.8:", delta_wilcoxon_0.8, "\n")

# Wykres mocy testów
ggplot(df, aes(x = delta, y = power, color = test)) +
  geom_point(size = 2) +
  geom_hline(yintercept = 0.8, linetype = "dashed", color = "red") +
  labs(title = "Moc testów t-Studenta i Wilcoxona",
       x = expression(Delta),
       y = "Moc",
       color = "Test") +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.title = element_text(hjust = 0.5)) +
  scale_color_manual(values = c("blue", "green"))