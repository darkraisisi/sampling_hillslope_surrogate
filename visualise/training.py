# Plot the MSE history of the training
plt.figure()
for key in history.history.keys():
    plt.plot(history_df[key], label=key)
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Custom MSE')