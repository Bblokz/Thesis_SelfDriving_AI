import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def read_loss_data(file_path):
    """Reads loss data from the given file path."""
    losses = []
    test_names = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Test") or line.startswith("C"):
                parts = line.split(':')
                test_name = parts[0].strip()
                loss = float(parts[1].split()[3])
                losses.append(loss)
                test_names.append(test_name)
            elif line.startswith("OVERALL"):
                break
    return test_names, losses


def plot_loss_graph(regular_losses, depth_losses, pure_depth_losses, test_names, filename):
    """Plots a graph for the loss data with advanced styling."""
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    # Creating a plot
    ax = plt.subplot(111)

    # Plotting with thicker lines and filled areas
    l1 = ax.fill_between(test_names, regular_losses, color=[.5, .5, .8, .3], edgecolor=[0, 0, .5, .3], linewidth=3)
    l2 = ax.fill_between(test_names, depth_losses, color=[.8, .5, .5, .3], edgecolor=[.5, 0, 0, .3], linewidth=3)
    l3 = ax.fill_between(test_names, pure_depth_losses, color=[.5, .8, .5, .3], edgecolor=[0, .5, 0, .3], linewidth=3)

    # Set basic properties
    ax.set_xlabel('Test Run', fontsize=12, style='italic')
    ax.set_ylabel('Loss', fontsize=12, style='italic')
    ax.set_title('Loss Comparison among RGB Only, Depth & RGB, and Depth Only Models', fontsize=14, weight='bold')

    # Set limits
    ax.set_xlim(0, len(test_names)-1)
    ax.set_ylim(min(min(regular_losses), min(depth_losses), min(pure_depth_losses)), max(max(regular_losses), max(depth_losses), max(pure_depth_losses)))

    # Customizing ticks
    ax.set_xticks(np.arange(len(test_names)))
    ax.set_xticklabels(test_names, rotation=45)
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)

    # Change color of the top and right spines
    ax.spines['right'].set_color((.8, .8, .8))
    ax.spines['top']. set_color((.8, .8, .8))

    # Removing unwanted spines
    sns.despine()

    # Adding legend
    plt.legend([l1, l2, l3], ['RGB Only Model', 'Depth & RGB Model', 'Depth Only Model'], loc='upper right')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()



def read_accuracy_data(file_path, tolerance_section=0):
    """Reads throttle and steering accuracy data from the given file path."""
    throttle_accuracies = []
    steering_accuracies = []
    sign_accuracies = []
    test_names = []
    section_count = 0
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("RunFolder OVERALL"):
                section_count += 1
                continue

            if section_count == tolerance_section and (line.startswith("Test") or line.startswith("C")):
                parts = line.split(':')
                test_name = parts[0].strip()
                values = parts[1].split()
                throttle_accuracy = float(values[0])
                steering_accuracy = float(values[1])
                sign_accuracy = float(values[2])

                throttle_accuracies.append(throttle_accuracy)
                steering_accuracies.append(steering_accuracy)
                sign_accuracies.append(sign_accuracy)
                test_names.append(test_name)

    return test_names, throttle_accuracies, steering_accuracies, sign_accuracies

def plot_throttle_accuracy_graph(test_names, rgb_throttle, depth_throttle, pure_depth_throttle, extra_throttle, file_name):
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")

    # Creating a plot
    ax = plt.subplot(111)

    # Plotting with thicker lines
    ax.plot(test_names, rgb_throttle, label='Red Altered Throttle', color='red', linewidth=3)
    ax.plot(test_names, depth_throttle, label='Blue Altered Throttle', color='blue', linewidth=3)
    ax.plot(test_names, pure_depth_throttle, label='Green Altered Throttle', color='green', linewidth=3)
    ax.plot(test_names, extra_throttle, label='Unalterd RGB', color='black', linewidth=3)

    # Set basic properties
    ax.set_xlabel('Test Run', fontsize=12, style='italic')
    ax.set_ylabel('Throttle Accuracy (%)', fontsize=12, style='italic')
    ax.set_title('Throttle Accuracy Comparison', fontsize=14, weight='bold')

    # Set limits and customizing ticks
    ax.set_xlim(0, len(test_names)-1)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(len(test_names)))
    ax.set_xticklabels(test_names, rotation=45)
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)

    # Change color of the top and right spines
    ax.spines['right'].set_color((.8,.8,.8))
    ax.spines['top'].set_color((.8,.8,.8))

    # Removing unwanted spines
    sns.despine()

    # Adding legend
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{file_name}_throttle.png')
    plt.show()

def plot_steering_accuracy_graph(test_names, rgb_steering, depth_steering, pure_depth_steering, rgb_sign, depth_sign, pure_depth_sign, extra_steering, extra_sign, file_name):
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")

    # Creating a plot
    ax = plt.subplot(111)

    # Plotting with thicker lines and dashed lines for sign accuracy
    # ax.plot(test_names, rgb_steering, label='Red Altered Steering', color='red', linewidth=3)
    # ax.plot(test_names, depth_steering, label='Blue Altered Steering', color='blue', linewidth=3)
    # ax.plot(test_names, pure_depth_steering, label='Green Altered Steering', color='green', linewidth=3)
    # ax.plot(test_names, extra_steering, label='Unaltered RGB', color='black', linewidth=3)
    ax.plot(test_names, extra_sign, label='Unaltered RGB Steering Sign', color='black', linestyle='dashed', linewidth=3)
    ax.plot(test_names, rgb_sign, label='Red Altered Steering', color='red', linestyle='dashed', linewidth=3)
    ax.plot(test_names, depth_sign, label='Blue Altered Steering Sign', color='blue', linestyle='dashed', linewidth=3)
    ax.plot(test_names, pure_depth_sign, label='Green Altered Steering Sign', color='green', linestyle='dashed', linewidth=3)

    # Set basic properties
    ax.set_xlabel('Test Run', fontsize=12, style='italic')
    ax.set_ylabel('Steering Accuracy (%)', fontsize=12, style='italic')
    ax.set_title('Steering Accuracy Comparison', fontsize=14, weight='bold')

    # Set limits and customizing ticks
    ax.set_xlim(0, len(test_names)-1)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(len(test_names)))
    ax.set_xticklabels(test_names, rotation=45)
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)

    # Change color of the top and right spines
    ax.spines['right'].set_color((.8,.8,.8))
    ax.spines['top'].set_color((.8,.8,.8))

    # Removing unwanted spines
    sns.despine()

    # Adding legend
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{file_name}_steering.png')
    plt.show()


def read_avg_abs_diff_data(file_path):
    """Reads average absolute difference data for throttle and steering from the given file path."""
    avg_abs_diff_throttle = []
    avg_abs_diff_steering = []
    test_names = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Test") or line.startswith("C"):
                parts = line.split(':')
                test_name = parts[0].strip()
                values = parts[1].split()
                avg_throttle_diff = float(values[4])
                avg_steering_diff = float(values[5])

                avg_abs_diff_throttle.append(avg_throttle_diff)
                avg_abs_diff_steering.append(avg_steering_diff)
                test_names.append(test_name)
            elif line.startswith("OVERALL"):
                break
    return test_names, avg_abs_diff_throttle, avg_abs_diff_steering

def plot_avg_abs_diff_graph(test_names, regular_throttle, regular_steering, depth_throttle, depth_steering, pure_depth_throttle, pure_depth_steering, filename):
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    ax = plt.subplot(111)

    # Plotting the average absolute differences for each model
    ax.plot(test_names, regular_throttle, label='RGB Only Throttle', color='red', linewidth=3)
    ax.plot(test_names, regular_steering, label='RGB Only Steering', color='orange', linewidth=3)
    ax.plot(test_names, depth_throttle, label='Depth & RGB Throttle', color='blue', linewidth=3)
    ax.plot(test_names, depth_steering, label='Depth & RGB Steering', color='lightblue', linewidth=3)
    ax.plot(test_names, pure_depth_throttle, label='Depth Only Throttle', color='green', linewidth=3)
    ax.plot(test_names, pure_depth_steering, label='Depth Only Steering', color='lime', linewidth=3)

    # Setting graph properties
    ax.set_xlabel('Test Run', fontsize=12, style='italic')
    ax.set_ylabel('Average Absolute Difference', fontsize=12, style='italic')
    ax.set_title('Average Absolute Difference in Throttle and Steering', fontsize=14, weight='bold')

    ax.set_xlim(0, len(test_names)-1)
    ax.set_ylim(0, max(max(regular_throttle), max(regular_steering), max(depth_throttle), max(depth_steering), max(pure_depth_throttle), max(pure_depth_steering)))

    ax.set_xticks(np.arange(len(test_names)))
    ax.set_xticklabels(test_names, rotation=45)
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)

    ax.spines['right'].set_color((.8,.8,.8))
    ax.spines['top'].set_color((.8,.8,.8))

    sns.despine()

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def main():
    # Paths to the text files
    regular_file_path = './regular.txt'
    depth_file_path = './depth.txt'
    pure_depth_file_path = './pureDepth.txt'
    extra_file_path = './extra.txt'

    while True:
        print("\nGraph Generation Menu:")
        print("1. Generate All Graphs")
        print("2. Generate Loss Graph")
        print("3. Generate Accuracy Graph (5% Tolerance)")
        print("4. Generate Accuracy Graph (10% Tolerance)")
        print("5. Generate Average Absolute Difference Graph")
        print("6. Quit")

        choice = input("Enter your choice (1-6): ")

        if choice == '1':
            generate_all_graphs(regular_file_path, depth_file_path, pure_depth_file_path)
        elif choice == '2':
            generate_loss_graph(regular_file_path, depth_file_path, pure_depth_file_path)
        elif choice == '3':
            generate_accuracy_graphs(regular_file_path, depth_file_path, pure_depth_file_path, extra_file_path, 0)
        elif choice == '4':
            generate_accuracy_graphs(regular_file_path, depth_file_path, pure_depth_file_path, 1)
        elif choice == '5':
            generate_avg_abs_diff_graph(regular_file_path, depth_file_path, pure_depth_file_path)
        elif choice == '6':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

def generate_all_graphs(regular_file_path, depth_file_path, pure_depth_file_path):
    generate_loss_graph(regular_file_path, depth_file_path, pure_depth_file_path)
    generate_accuracy_graphs(regular_file_path, depth_file_path, pure_depth_file_path, 0)
    generate_accuracy_graphs(regular_file_path, depth_file_path, pure_depth_file_path, 1)
    generate_avg_abs_diff_graph(regular_file_path, depth_file_path, pure_depth_file_path)


def generate_loss_graph(regular_file_path, depth_file_path, pure_depth_file_path):
    regular_test_names, regular_losses = read_loss_data(regular_file_path)
    depth_test_names, depth_losses = read_loss_data(depth_file_path)
    pureDepth_test_names, pureDepth_losses = read_loss_data(pure_depth_file_path)
    loss_file_name = '../Graphs/loss.png'
    plot_loss_graph(regular_losses, depth_losses, pureDepth_losses, regular_test_names, loss_file_name)

def generate_accuracy_graphs(regular_file_path, depth_file_path, pure_depth_file_path, extra_file_path, section):
    # Reading data for all models
    regular_test_names, regular_throttle, regular_steering, regular_sign = read_accuracy_data(regular_file_path, section)
    depth_test_names, depth_throttle, depth_steering, depth_sign = read_accuracy_data(depth_file_path, section)
    pure_depth_test_names, pure_depth_throttle, pure_depth_steering, pure_depth_sign = read_accuracy_data(pure_depth_file_path, section)
    extra_test_names, extra_throttle, extra_steering, extra_sign = read_accuracy_data(extra_file_path, section)

    # Generating throttle accuracy graph
    throttle_file_name = f'../Graphs/throttle_accuracy_comparison_{section*5+5}percent.png'
    plot_throttle_accuracy_graph(regular_test_names, regular_throttle, depth_throttle, pure_depth_throttle, extra_throttle, throttle_file_name)

    # Generating steering accuracy graph
    steering_file_name = f'../Graphs/steering_accuracy_comparison_{section*5+5}percent.png'
    plot_steering_accuracy_graph(regular_test_names, regular_steering, depth_steering, pure_depth_steering, regular_sign, depth_sign, pure_depth_sign, extra_steering, extra_sign, steering_file_name)

def generate_avg_abs_diff_graph(regular_file_path, depth_file_path, pure_depth_file_path):
    regular_test_names, regular_avg_abs_diff_throttle, regular_avg_abs_diff_steering = read_avg_abs_diff_data(regular_file_path)
    depth_test_names, depth_avg_abs_diff_throttle, depth_avg_abs_diff_steering = read_avg_abs_diff_data(depth_file_path)
    pure_test_names, pure_throttle, pure_steering = read_avg_abs_diff_data(pure_depth_file_path)
    avg_abs_diff_file_name = '../Graphs/avg_abs_diff.png'
    plot_avg_abs_diff_graph(regular_test_names,
                                regular_avg_abs_diff_throttle, regular_avg_abs_diff_steering,
                                depth_avg_abs_diff_throttle, depth_avg_abs_diff_steering,
                                pure_throttle, pure_steering,
                                avg_abs_diff_file_name)

if __name__ == "__main__":
    main()
