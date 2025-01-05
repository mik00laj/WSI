set.seed(123)

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

# Obliczenie mocy testu Wilcoxona
power_wilcoxon_test <- calculate_power_wilcoxon_test(n, delta, sigma, alpha, num_simulations)

# Wykres mocy testu Wilcoxona
plot(delta, power_wilcoxon_test, type = "b", col = "blue", pch = 19,
     xlab = expression(delta), ylab = "Power",
     main = "Power of Wilcoxon Test",
     ylim = c(0, 1))
abline(h = 0.8, col = "red", lty = 2)
legend("bottomright", legend = c("Wilcoxon Test", "80% Power"), col = c("blue", "red"), lty = c(1, 2), pch = c(19, NA))
