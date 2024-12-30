from matplotlib import pyplot as plt

x = [i+1 for i in range(20)]






y_0 = [93.97,97.05,97.7,98.77,98.26,98.36,98.76,98.87,98.90,98.88,98.97,98.68,98.95,99.17,98.92,99.00,99.12,98.94,99.03,98.78]
y_1 = [98.19, 98.50, 98.63, 99.15, 99.10, 99.20, 99.21, 99.29, 99.24, 98.80, 99.53, 99.53, 99.54, 99.49, 99.53, 99.46, 99.48, 99.43, 99.52, 99.49]
y_2 = [98.57, 98.90, 98.94, 99.01, 98.99, 99.10, 98.94, 98.35, 99.07, 98.82, 99.38, 99.39, 99.37, 99.38, 99.45, 99.40, 99.43, 99.38, 99.45, 99.41]

#FNet0
y_3 = [98.57, 98.90, 98.94, 99.01, 98.99, 99.10, 98.94, 98.35, 99.07, 98.82, 99.38, 99.39, 99.37, 99.38, 99.45, 99.40, 99.43, 99.38, 99.45, 99.41]

#FNet1
y_4 = [98.46, 98.24, 98.51, 99.25, 98.74, 99.20, 99.26, 99.32, 99.35, 99.35, 99.48, 99.51, 99.52, 99.53, 99.54, 99.58, 99.54, 99.58, 99.57, 99.56]

#FNet2
y_5 = [98.23, 99.05, 99.04, 99.10, 99.37, 99.04, 99.30, 98.93, 99.17, 98.93, 99.52, 99.53, 99.56, 99.56, 99.55, 99.52, 99.56, 99.55, 99.52, 99.55]

#FNet3
y_6 = [98.9, 98.86, 99.08, 98.75, 99.27, 99.04, 99.42, 99.35, 99.44, 99.25, 99.58, 99.62, 99.6, 99.62, 99.6, 99.61, 99.6, 99.64, 99.6, 99.65]

# Time for the models
text0 = 261.38
text1 = 237.56
text2 = 238.93

#pic 3
text3 = 238.93  #FNet0
text4 = 256.66  #FNet1
text5 = 259.51  #FNet2
text6 = 286.44  #FNet3
# Draw the plot
plt.figure(figsize=(10, 5))
plt.plot(x, y_4, label='FNet1')
plt.plot(x, y_6, label='FNet3')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy of FNet')
plt.legend()
plt.grid()

# Force x-axis to show integers from 1 to 20
ax = plt.gca()
ax.set_xticks(x)  # Explicitly set ticks to be 1 to 20

# Add annotations
plt.text(
    0.95, 0.99, 
    f'FNet1 spent: {text4:.2f}s', 
    transform=ax.transAxes, 
    ha='right', 
    va='top',
    fontsize=12,    # Font size adjustable
)
plt.text(
    0.95, 0.9, 
    f'FNet3 spent: {text6:.2f}s', 
    transform=ax.transAxes, 
    ha='right', 
    va='top',
    fontsize=12,    # Font size adjustable
)

# Save the figure
path = './5_FNET13.png'
plt.savefig(path)
print(path)