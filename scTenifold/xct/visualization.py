from __future__ import annotations

from collections import namedtuple

import igraph as ig
import numpy as np
import scipy.sparse


def get_Xct_pairs(df) -> tuple:
    return tuple(n.split('_') for n in list(df.index))


Style = namedtuple("visual_style", ["vertex_size",
                                    "vertex_label_size",
                                    "vertex_label_dist",
                                    "edge_curved",
                                    "margin",
                                    "layout",
                                    "bbox_size",
                                    "gene_color",
                                    "gene_shape",
                                    "tf_color",
                                    "tf_shape", "mark_color"],
                   defaults=(50, 12, 0.0, 0.1, 70, "large", (512, 512),
                             "darkorange", "square", "lightgray", "circle", ["whitesmoke", "whitesmoke"]))


visual_style_pcnet = Style(layout="large", bbox_size=(512, 512))
visual_style_xnet = Style(layout="kk", bbox_size=(768, 768))


def _interact_graph(main_graph: ig.Graph, matched_g: ig.Graph):
    if not isinstance(matched_g, ig.Graph):
        raise TypeError('match vertices to an igraph object')
    v_names_to_match = list(set(main_graph.vs['name']).intersection(set(matched_g.vs['name'])))
    to_delete_ids = []
    for v in main_graph.vs:
        if v['name'] not in v_names_to_match:
            to_delete_ids.append(v.index)
    main_graph.delete_vertices(to_delete_ids)
    return main_graph


def _parse_kws(kws, g, bbox_scale, g1=None, g2=None,
               edge_width_scale = None, edge_width_max = None, add_edges = None):
    kws = {k: v for k, v in kws.items()}
    bbox_size = kws.pop("bbox_size")
    tf_color = kws.pop("tf_color")
    tf_shape = kws.pop("tf_shape")
    gene_color = kws.pop("gene_color")
    gene_shape = kws.pop("gene_shape")
    mark_color = kws.pop("mark_color")
    kws["bbox"] = (bbox_size[0] * bbox_scale, bbox_size[1] * bbox_scale)
    kws["vertex_label"] = g.vs["name"]
    kws["vertex_color"] = [tf_color if tf else gene_color for tf in g.vs["is_TF"]]
    kws["vertex_shape"] = [tf_shape if tf else gene_shape for tf in g.vs["is_TF"]]
    if edge_width_scale is None:
        edge_width_scale = 3 / max(np.abs(g.es['weight']))

    if edge_width_max is not None:
        kws["edge_width"] = [ edge_width_scale * abs(w) if edge_width_scale * abs(w) < edge_width_max else edge_width_max
                              for w in g.es['weight']]
    else:
        kws["edge_width"] = [edge_width_scale * abs(w) for w in g.es['weight']]
    added_e = add_edges if add_edges is not None else 0
    kws["edge_color"] = [
        'red' if all([w > 0, i < len(g.es) - added_e])
        else 'blue' if i < len(g.es) - added_e
        else 'darkgreen'
        for i, w in enumerate(g.es['weight'])
    ]
    kws["edge_arrow_size"] = [1e-12 if i < len(g.es) - added_e else 1 for i, _ in enumerate(g.es)]
    if g1 is not None and g2 is not None:
        kws["mark_groups"] = (
            [(list(range(0, len(g1.vs))), mark_color[0])]
            + [(list(range(len(g1.vs), len(g1.vs) + len(g2.vs))), mark_color[1])]
        )
    else:
        kws["mark_groups"] = [(list(range(0, len(g.vs))), mark_color[0])]
    return kws


def plot_pcNet_method(net,
                      gene_names,
                      tf_names,
                      matched_graph=None,
                      top_edges=20,
                      remove_isolated_nodes=True,
                      bbox_scale=1,
                      show=True,
                      file_name=None,
                      verbose=False,
                      edge_width_scale=None,
                      **kwargs):
    subnet = net.subset_in(gene_names+tf_names, copy=True)
    # print('zeros after filter TFs:', subnet._net.shape, np.sum(subnet._net == 0))
    subnet.set_rows_and_cols_as(tf_names, 0)
    # print('zeros after set TFs as 0:', subnet._net.shape, np.sum(subnet._net == 0))
    # subnet.set_rows_as(tf_names, 0)
    # subnet.set_cols_as(tf_names, 0)

    g = ig.Graph.Weighted_Adjacency(scipy.sparse.tril(subnet.net), mode='directed', attr="weight",
                                    loops=True)  # upper triangular for directionality
    g.vs["name"] = subnet.gene_names
    g.vs["is_TF"] = subnet.gene_names.isin(tf_names)
    # print('g.es.weight, non-zero/all:', np.sum(np.array(g.es['weight']) != 0), '/', len(g.es['weight']), )
    if len(g.es) == 0:
        gene_names = ' '.join(map(str, gene_names))  # string
        raise ValueError(f'target gene {gene_names} generated 0 edge...')

    if matched_graph is not None:
        g = _interact_graph(g, matched_graph)

    # trim low weight edges
    if top_edges is not None:
        abs_weight = [abs(w) for w in g.es['weight']]
        edges_delete_ids = sorted(range(len(abs_weight)), key=lambda k: abs_weight[k])[:-top_edges]  # idx from bottom
        g.delete_edges(edges_delete_ids)

    if remove_isolated_nodes:
        to_delete_ids = []
        for v in g.vs:
            if v.degree() == 0:
                to_delete_ids.append(v.index)
                if v['name'] in gene_names:
                    print(f"{v['name']} has been removed due to degree equals to zero among top weighted edges")
        g.delete_vertices(to_delete_ids)

    if verbose:
        print(f'undirected graph constructed: \n# of nodes: {len(g.vs)}, # of edges: {len(g.es)}\n')

    kws = dict(visual_style_pcnet._asdict())
    kws.update(kwargs)
    kws = _parse_kws(kws, g, bbox_scale, edge_width_scale=edge_width_scale)

    if file_name is not None and verbose:
        print(f'graph saved as \"{file_name}\"')
    if show:
        return ig.plot(g, file_name, **kws)
    else:
        return g


def plot_XNet(g1, g2,
              gene_pairs, df_enriched=None, file_name: str | None = None,
              verbose: bool = False,
              edge_width_scale=None, edge_width_max: int = 5,
              bbox_scale: int = 1,
              show: bool = False,
              **kwargs):
    '''visualize merged GRN from sender and receiver cell types,
        use edge_width_scale to make two graphs width comparable (both using absolute values)'''
    gg = g1.disjoint_union(g2)  # merge disjointly
    if verbose:
        print(f'graphs merged: \n# of nodes: {len(gg.vs)}, # of edges: {len(gg.es)}\n')

    added_e = 0
    for pair in gene_pairs:
        edges_idx = (gg.vs.find(name=pair[0]).index, gg.vs.find(name=pair[1]).index)  # from to
        if df_enriched is None:
            w = 1.20
        else:
            if 'correspondence' not in df_enriched.columns:
                raise IndexError('require resulted dataframe with column \'correspondence\'')
            else:
                w_list = np.asarray(df_enriched['correspondence'])
                w_list = w_list * 1.20 / max(w_list)
                w = w_list[get_Xct_pairs(df_enriched).index(pair)]
        gg.add_edge(edges_idx[0], edges_idx[1], weight=w)  # set weight > 1 and be the max (default: 1.02)
        added_e += 1  # source and target both found
        if verbose:
            print(f'edge from {pair[0]} to {pair[1]} added')

    kws = dict(visual_style_xnet._asdict())
    kws.update(kwargs)
    kws = _parse_kws(kws, g=gg, g1=g1, g2=g2,
                     bbox_scale=bbox_scale,
                     edge_width_scale=edge_width_scale,
                     edge_width_max=edge_width_max, add_edges=added_e)
    if file_name is not None and verbose:
        print(f'graph saved as \"{file_name}\"')
    if show:
        return ig.plot(gg, file_name, **kws)
    else:
        return gg
