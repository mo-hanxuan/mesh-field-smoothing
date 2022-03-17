from elementsBody import ElementsBody
import numpy as np
from facetDenseRegular import *


def horizons(obj, iele, inHorizon=None):
    """
        get the horizons of an element of this object
    """
    if not isinstance(obj, ElementsBody):
        raise ValidationErr("error, not isinstance(obj, ElementsBody)")
    if not hasattr(obj, "horizons"):
        obj.horizons = {}
    
    inHorizon = obj.inHorizon if inHorizon == None else inHorizon
    if iele not in obj.horizons:
        obj.horizons[iele] = obj.findHorizon(iele=iele, inHorizon=inHorizon)
    return obj.horizons[iele]


def horizonsFacets(obj, iele, inHorizon=None):
    if not isinstance(obj, ElementsBody):
        raise ValidationErr("error, not isinstance(obj, ElementsBody)")

    inHorizon = obj.inHorizon if inHorizon == None else inHorizon
    patchEles = horizons(obj, iele, inHorizon)

    facets = {}
    for facet in obj.outerFacetDic:
        ele = obj.outerFacetDic[facet]
        if ele in patchEles:
            facets[facet] = {}
    edges = set()
    for edge in obj.facialEdgeDic:
        for facet in obj.facialEdgeDic[edge]:
            if facet in facets:
                edges.add(edge)
                break
    
    return facets, edges