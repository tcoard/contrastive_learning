
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ==> 15_indel_lin_loss.txt <==
# best accuracy: 14.67

# ==> 30_indel_lin_loss.txt <==
# best accuracy: 18.00

# ==> 45_indel_lin_loss.txt <==
# best accuracy: 15.33


# ==> 15_sub_lin_loss.txt <==
# best accuracy: 16.00

# ==> 30_sub_lin_loss.txt <==
# best accuracy: 16.00

# ==> 45_sub_lin_loss.txt <==
# best accuracy: 14.00


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # fig.suptitle('Horizontally stacked subplots')



    indel = [14.67, 18.00, 15.33]
    ax1.plot(indel, marker='o', linestyle='--', color='r')
    ax1.set(xlabel="Percent of Sequence Changed")
    ax1.set(ylabel="Accuracy")
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set(title="Indel")
    ax1.set_xticks((0,1,2))
    ax1.set_xticklabels((15, 30, 45))

    sub = [16.00, 16.00, 14.00]
    ax2.plot(sub, marker='o', linestyle='--', color='r')
    ax2.set(title="Substitution")
    ax2.set(xlabel="Percent of Sequence Changed")
    ax2.set_xticks((0,1,2))
    ax2.set_xticklabels((15, 30, 45))
    # ax2.xlabel("Percent of Sequence Changed")
    # ax2.xticks((0,1,2), (0.15, 0.30, 0.45))

    plt.savefig("test.png", bbox_inches="tight")
    plt.clf()

if __name__ == "__main__":
    main()

