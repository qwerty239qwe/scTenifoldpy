QuickStart
-------------

To use scTenifoldNet, initialize a scTenifoldNet object and execute build() method:

.. code-block:: python

    from scTenifold import get_test_df
    from scTenifold import scTenifoldNet

    df_1, df_2 = get_test_df(n_cells=1000), get_test_df(n_cells=1000)
    sc = scTenifoldNet(df_1, df_2, "X", "Y",
                       qc_kws={"min_lib_size": 10})
    result = sc.build()

To use scTenifoldKnk, initialize a scTenifoldKnk object and execute build() method:

.. code-block:: python

    from scTenifold import get_test_df
    from scTenifold import scTenifoldKnk

    df = get_test_df(n_cells=1000)
    sc = scTenifoldKnk(data=df,
                       ko_method="default",
                       ko_genes=["NG-1"],  # the gene you wants to knock out
                       qc_kws={"min_lib_size": 10})
    result = sc.build()
