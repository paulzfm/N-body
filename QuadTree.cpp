#include "QuadTree.h"

void QuadTree::build(const std::vector<Point>& points)
{

}

void QuadTree::insert(const Body& body, int node)
{
    if (_nodes[node].leaf) { // is leaf node
        _nodes[node].center = body.center;
        _nodes[node].mass = body.mass;
        _nodes[node].leaf = false;
        return;
    }

    if (!_nodes[node].expanded) { // is external node
        for (int i = 0; i < 4; i++) {
            _nodes[next].rect = _nodes[node].rect.divide(i);
            _nodes[node].children[i] = _next++;
        }
        expanded = true;

        Body tmp;
        tmp.center = _nodes[node].center;
        tmp.mass = _nodes[node].mass;
        insert(body, node);
        insert(tmp, node);

        _nodes[node].update(body);
        _nodes[node].update(tmp);
        return;
    }

    // is internal node
    _nodes[node].update(body);
    for (int i = 0; i < 4; i++) {
        int child = _nodes[node].children[i];
        if (body.pos.inside(_nodes[child])) {
            insert(body, child);
            return;
        }
    }
}
