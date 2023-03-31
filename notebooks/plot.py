""" Inventaire des fonctions d'affichage graphique :

pepper_utils.py :
* plot_discrete_stats
* show_discrete_stats (graphique + dataframe)

im_ml.py
* select_k_with_anova : la partie plot devrait être séparée
* select_k_with_davies_bouldin : idem
* show_tsne

tx_ml.py
* show_sparsity
* show_tsne

flipkart_utils.py
* display_product_gallery


tx_prep
* show_lens_dist
* show_lexicon_dist

...

"""

import matplotlib.pyplot as plt



"""def titles_labels():
    # plot_discrete_stats
    plt.yscale('log')
    plt.ylim(precision, 100)

    # Axis titles
    plt.ylabel('Filling rate & Shannon entropy')
    plt.xlabel('')

    # Rotate x-axis labels
    plt.xticks(rotation=30, ha='right')

    # Add overall title
    plt.title(f'Discrete statistics of `{table_name}` table', fontsize=16)


    # select_k_with_anova (à séparer en 2 fonctions)
    plt.xlabel('Number of clusters')
    plt.xticks(k_values)
    if metric == 'inertia':
        plt.ylabel('Inertia')
    elif metric == 'silhouette':
        plt.ylabel('Silhouette Score')
    norm_status = ', not scaled'
    if normalize:
        norm_status = ', scaled'
    plt.title(f'ANOVA with Elbow Method ({metric}{norm_status})', weight='bold')


    # select_k_with_davies_bouldin (à séparer en 2 fonctions)
    plt.xlabel("Number of clusters")
    plt.ylabel("Davies-Bouldin index")
    norm_status = ', not scaled'
    if normalize:
        norm_status = ', scaled'
    plt.title(f'Davis-Bouldin index{norm_status}', weight='bold')

    # show_tsne (im)
    plt.title(f"TSNE against {labels_type}", fontsize=30, pad=35, fontweight="bold")
    plt.xlabel("tsne_0", fontsize=26, fontweight="bold")
    plt.ylabel("tsne_1", fontsize=26, fontweight="bold")
    plt.legend(prop={"size": 14}) 

    # show_tsne (tx)
    plt.title(f"{what} by true classes", weight="bold", pad=10)
    plt.title(f"{what} by clusters", weight="bold", pad=10)

    # show_lens_dist
    plt.xlabel(f"Number of {elts_type}")
    plt.title(
        f"`{lens.name}` {clip} {seq_type} length distribution",
        weight="bold", pad=10
    )

    # show_lexicon_dist
    var_name = lexicon_data.index.name  # {clip}
    feat_name = 'length' if feature == 'len' else 'frequency'
    plt.xlabel(f"Word {feat_name}")
    plt.title(
        f"`{var_name}` {clip} word {feat_name} distribution",
        weight="bold", pad=10
    )"""