CLASS_LOOKUP = [
    "AMINOGLYCOSIDE",
    "BETA-LACTAM",
    "FOLATE-SYNTHESIS-INHABITOR",
    "GLYCOPEPTIDE",
    "MACROLIDE",
    "MULTIDRUG",
    "PHENICOL",
    "QUINOLONE",
    "TETRACYCLINE",
    "TRIMETHOPRIM",
    ]

def main():
    cf_matrix = [[1, 1, 5, 0, 0, 1, 4, 1, 0, 2],
    [4, 1, 4, 0, 1, 1, 2, 2, 0, 0],
    [1, 3, 6, 0, 1, 1, 1, 1, 0, 1],
    [4, 1, 3, 0, 1, 0, 5, 1, 0, 0],
    [5, 2, 2, 0, 1, 0, 3, 1, 0, 1],
    [1, 1, 0, 0, 2, 5, 0, 5, 0, 1],
    [0, 0, 6, 0, 0, 1, 3, 2, 1, 2],
    [1, 2, 4, 0, 0, 0, 2, 5, 1, 0],
    [1, 1, 6, 0, 1, 0, 1, 0, 2, 3],
    [1, 1, 3, 0, 0, 0, 4, 3, 0, 3]]


    # cf_matrix = [[0, 0, 0, 0, 12, 2, 0, 0, 1, 0],
# [0, 0, 0, 2, 8, 1, 0, 0, 4, 0],
# [0, 0, 0, 2, 5, 3, 0, 0, 5, 0],
# [0, 0, 0, 2, 7, 4, 0, 0, 2, 0],
# [0, 0, 0, 4, 9, 1, 0, 0, 1, 0],
# [0, 0, 0, 1, 4, 5, 0, 0, 5, 0],
# [0, 0, 0, 5, 2, 4, 0, 0, 4, 0],
# [0, 0, 0, 2, 5, 5, 0, 0, 3, 0],
# [0, 0, 0, 2, 5, 2, 0, 0, 6, 0],
# [0, 0, 0, 2, 7, 5, 0, 0, 1, 0]]
 #   * Acc@1 14.667  



    import seaborn as sns
    heat = sns.heatmap(cf_matrix, xticklabels=CLASS_LOOKUP, yticklabels=CLASS_LOOKUP, annot=True)
    heat.set_title('ESM Confusion Matrix')
    heat.figure.tight_layout()
    fig = heat.get_figure()
    fig.savefig('conf_mat.png')

if __name__ == "__main__":
    main()

