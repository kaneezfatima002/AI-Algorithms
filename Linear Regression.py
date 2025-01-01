def avg(data):
    total = 0
    for d in data:
        total += d
    return total / len(data)

def find_slope(x, y, avg_x, avg_y):
    num = 0
    den = 0
    for i in range(len(x)):
        num += (x[i] - avg_x) * (y[i] - avg_y)
        den += (x[i] - avg_x) ** 2
    return num / den

def find_intercept(avg_x, avg_y, m):
    return avg_y - m * avg_x

def make_predictions(x, b0, b1):
    results = []
    for val in x:
        results.append(b0 + b1 * val)
    return results

def find_mse(actual, predicted):
    total = 0
    for i in range(len(actual)):
        total += (actual[i] - predicted[i]) ** 2
    return total / len(actual)

def optimize_weights(x, y, b0, b1, lr, steps):
    for _ in range(steps):
        grad_b0 = 0
        grad_b1 = 0
        for i in range(len(x)):
            pred = b0 + b1 * x[i]
            error = y[i] - pred
            grad_b0 += -2 * error
            grad_b1 += -2 * error * x[i]
        grad_b0 /= len(x)
        grad_b1 /= len(x)
        b0 -= lr * grad_b0
        b1 -= lr * grad_b1
    return b0, b1

def train_model(x, y, lr=0.01, steps=1000):
    avg_x = avg(x)
    avg_y = avg(y)
    m = find_slope(x, y, avg_x, avg_y)
    b = find_intercept(avg_x, avg_y, m)
    b0, b1 = optimize_weights(x, y, b, m, lr, steps)
    return b0, b1

def run_test():
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000]
    lr = 0.01
    steps = 1000
    b0, b1 = train_model(x, y, lr, steps)
    preds = make_predictions(x, b0, b1)
    error = find_mse(y, preds)
    print("Slope:", b1)
    print("Intercept:", b0)
    print("MSE:", error)
    print("Predictions:", preds)

run_test()

