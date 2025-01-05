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

# Parametry
n <- 30
sigma <- 1
alpha <- 0.05
num_simulations <- 1000
delta <- seq(-2, 2, by = 0.1)

# Obliczenie mocy testu t-Studenta
power_t_test <- calculate_power_t_test(n, delta, sigma, alpha, num_simulations)

# Wykres mocy testu t-Studenta
plot(delta, power_t_test, type = "b", col = "blue", pch = 19,
     xlab = expression(delta), ylab = "Power",
     main = "Power of t-test",
     ylim = c(0, 1))
abline(h = 0.8, col = "red", lty = 2)
legend("bottomright", legend = c("t-test", "80% Power"), col = c("blue", "red"), lty = c(1, 2), pch = c(19, NA))

