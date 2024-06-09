// bipartite check --> Graph should contain cycles of even length only
class Solution
{
public:
    bool dfs(int node, vector<vector<int>> &g, vector<int> &color)
    {

        if (color[node] == -1)
        {
            color[node] = 1;
        }

        for (auto it : g[node])
        {
            if (color[it] == -1)
            {
                color[it] = 1 - color[node];
                if (dfs(it, g, color) == 0)
                    return 0;
            }
            else if (color[node] == color[it])
            {
                return 0;
            }
        }

        //condition when the graph is bipartite
        return 1;
    }

    bool isBipartite(vector<vector<int>> &graph)
    {
        int n = graph.size();

        vector<vector<int>> g(n);
        for (auto it : graph)
        {
            vector<int> temp = it;
            if (temp.size() == 0)
            {
                continue;
            }
            int u = temp[0];
            for (int i = 1; i < temp.size(); i++)
            {
                g[u].push_back(temp[i]);
            }
        }

        vector<int> color(n, -1);

        for (int i = 0; i < n; i++)
        {
            if (color[i] == -1)
            {
                bool f = dfs(i, g, color);
                if (f == 0)
                    return 0;
            }
        }

        return 1;
    }
};

// Cycle detection in undirected graph

bool sycle(ll node, ll par)
{
    vis[node] = 1;
    for (auto child : adj[node])
    {
        if (vis[child] == 0)
        {
            if (dfs(child, node) == 1)
            {
                return 1;
            }
        }
        else
        {
            if (child != par)
            {
                return 1;
            }
        }
    }

    return 0;
}

// Cycle detection in Directed graph

int dfs(int node, vector<vector<int>> &ar, vector<int> &vis, vector<int> &pathvis)
{
    vis[node] = 1;
    pathvis[node] = 1;

    for (auto child : ar[node])
    {
        if (vis[child] == 0)
        {
            if (dfs(child, ar, vis, pathvis) == 1)
            {
                return 1;
            }
        }
        else if (pathvis[child] == 1)
        {
            return 1;
        }
    }
    pathvis[node] = 0;
    return 0;
}

// Diameter of a Tree -> Run DFS two times to get this
// ll maxD(Stores max distance) , ll maxNd (stores the node at the maximum distance from the source node)
// First call - dia_tree(1,0)  --> Returns the maxND
// Second call - dia_tree(maxND, 0) --> Returns the maximum node
void dia_tree(ll node, ll dis)
{
    vis[node] = 1;

    if (d > maxD)
    {
        maxD = d;
        maxND = node;
    }

    for (auto child : adj[node])
    {
        if (vis[child] == 0)
        {
            dia_tree(child, d + 1);
        }
    }
}
// you have access to distance as well as farthest node
// count of nodes in each sub tree

// subtree size
void sub(ll node, ll par)
{
    for (auto it : g[node])
    {
        if (it != par)
        {
            sub(it, node);
            cnt[node] += cnt[it];
        }
    }
    return;
}

// In and out time for each node
ll timer = 0;
void time(ll node)
{
    // timer++;
    vis[node] = 1;
    in[node] = timer++;
    for (auto child : adj[node])
    {
        if (vis[child] == 0)
        {
            time(child);
        }
    }
    // timer++;
    ou[node] = timer++;
}

// Subtree - size == ((out - in)/2)  + 1

// Subtree size calculation - subsize[v] --> Stores the size of the subtree rooted at node v.
ll subtree(ll node)
{
    ll size = 1;
    vis[node] = 1;
    for (auto child : adj[node])
    {
        if (vis[child] == 0)
        {
            size += subtree(child);
        }
    }
    d[node] = size; // Storing the size of sub-tree
    return size;
}

// Stnadard BFS implementation --> Uses queue idea
void bfs(ll src)
{
    vis[src] = 1;
    dis[src] = 0;

    queue<ll> q;
    q.push(src);

    while (!q.empty())
    {
        ll curr = q.front();
        q.pop();

        for (auto child : adj[curr])
        {
            if (vis[child] == 0)
            {
                q.push(child);
                dist[child] = dist[curr] + 1;
                vis[child] = 1;
            }
        }
    }
}

// Problem on BFS
// find minimum moves required to get A from B
// when to can change one digit at a time
// only 4 digit prime numbers are involved

// Finding Cut Edges
// Finding Cut Vertices

// Topological sorting --> Separate indegree array should be
// maintained to check the in degree of nodes
// only for DAG (directed acyclic graph)
//  also see using
vl res; // the sorting is stored here
void kahn(ll n)
{
    // n - no. of vertices
    queue<int> q; // min-heap priority queue declaration ---> priority_queue<int, vector<int>, greater<int>>
    for (ll i = 1; i <= n; i++)
    {
        if (in[i] == 0)
        {
            q.push(i);
        }
    }

    while (!q.empty())
    {
        ll cur = q.front();
        q.pop();
        // res will contain the our topo sort
        res.push_back(cur);

        for (auto it : ar[cur])
        {
            in[it]--;
            if (in[it] == 0)
                q.push(it);
        }
    }
}

// DFS on 2D - Grid
ll dx[] = {-1, 0, 1, 0};
ll dy[] = {0, 1, 0, -1};
bool isValid(ll x, ll y)
{
    if (x < 1 or y < 1 or x > n or y > m)
        return 0;
    if (vis[x][y] == 1 or ar[x][y] == '#')
        (cannot be entered or something other condition) return 0;
    return 1;
}
void dfs(ll x, ll y)
{
    vis[x][y] = 1;
    for (ll i = 0; i < 4; i++)
    {
        if (isValid(x + dx[i], y + dy[i]))
            dfs(x + dx[i], y + dy[i]);
    }
}

// BFS on 2-D grid

ll dx[] = {-1, 0, 1, 0};
ll dy[] = {0, 1, 0, -1};
bool isValid(ll x, ll y)
{
    if (x < 1 or y < 1 or x > n or y > m)
        return 0;
    if (vis[x][y] == 1 or ar[x][y] == '#'(cannot be entered or somrthing other condition))
        return 0;
    return 1;
}

void BFS(ll srcX, ll srcY)
{
    vis[srcX][srcY] = 1;
    dis[srcX][srcY] = 0;

    queue<pair<int, int>> q;
    q.push({srcX, srcY});

    while (!(q.empty()))
    {
        int curx = q.front().first;
        int cury = q.front().second;
        q.pop();
        for (ll i = 0; i < 4; i++)
        {
            if (isValid(curx + dx[i], cury + dy[i]))
            {
                int newx = curx + dx[i];
                int newy = cury + dy[i];

                dist[newx][newy] = dis[curx][cury] + 1;
                vis[newx][newy] = 1;
                q.push({newx, newy});
            }
        }
    }
}

// Knight moves
ll dx[] = {2, 2, -2, -2, 1, 1, -1, -1}; // All plus , minus combinations are here with
ll dy[] = {1, -1, 1, -1, 2, -2, 2, -2};

/*Finding bridges in the graph  --> O(n+2*m)
tin[] - stores time of insertion during dfs
low[] - min of all adjacent nodes apart from parent*/
int timer = 1;
void dfs(int node, int parent, vector<int> &vis, vector<vector<int>> &g, vector<int> &tin, vector<int> & low, vector<vector<int>> &bridges){
    vis[node] = 1;
    tin[node] = low[node] = timer++;

    for(auto it: g[node]){
        if(it == parent) continue;
        if(!vis[it]){
            dfs(it, node, vis, g, tin, low, bridges);
            low[node] = min(low[node], low[it]);
             
             if(low[it] > tin[node]){
                 bridges.push_back({it, node});
             }
        }
        else{
            /*already visited before, so
            no bridge definitely*/
            low[node] = min(low[node], low[it]);
        }
    }
}
vector<vector<int>> criticalConnections(int n, vector<vector<int>>& connections) {
    vector<vector<int>> g(n);

    for(auto it: connections){
        g[it[0]].push_back(it[1]);
        g[it[1]].push_back(it[0]);
    }
    vector<int> vis(n, 0), tin(n), low(n);
    vector<vector<int>> bridges;
    dfs(0, -1, vis, g, tin, low, bridges);

    return bridges;
}

/*Finding Articulation points - on removal of these nodes, graph breaks into
two or more connected components
tin[] - stores time of insertion during dfs
low[] - min of all adjacent nodes apart from (parent and **visited nodes)
*/

    int timer = 1;
    void dfs(int node, int parent, vector<int> &vis, vector<int> & tin, vector<int> &low, vector<int> &mark, vector<int> adj[])
    {
        vis[node] = 1;
        
        tin[node] = low[node] = timer++;
        int child = 0;
        
        for(auto it: adj[node]){
            if(it == parent) continue;
            if(!vis[it]){
                dfs(it, node, vis, tin, low, mark, adj);
                low[node] = min(low[node], low[it]);
                if(low[it] >= tin[node] and parent != -1){
                    mark[node] = 1;
                }
                child++;
            }
            else{
                // difference from bridges
                low[node] = min(low[node], tin[it]);
            }
        }
        if(child > 1 and parent == -1){
            mark[node] = 1;
        }
    }
    vector<int> articulationPoints(int n, vector<int>adj[]) {
        // Code here
        vector<int> vis(n, 0), tin(n), low(n), mark(n,0);
        for(int i = 0; i < n; i++){
            if(!vis[i]){
                dfs(i, -1, vis, tin, low, mark, adj);
            }
        }
        vector<int> ans;
        for(int i = 0; i < n; i++){
            if(mark[i] == 1){
                ans.push_back(i);
            }
        }
        
        if(ans.size() == 0) return {-1};
        return ans;
    }


// minimum spanning tree - kruskals algorithm
// O(m logn)
void mst()
{
    ll n, m;
    cin >> n >> m;
    ll sum = 0;
    vector<pair<ll, pair<ll, ll>>> e;

    rep(i, 0, m)
    {
        ll u, v, w;
        cin >> u >> v >> w;
        e.pb({w, (u, v)});
    }

    sort(e.begin(), e.end());

    rep(i, 0, m)
    {
        ll u = e[i].S.F;
        ll v = e[i].S.S;

        if (findpar(u) != findpar(v))
        {
            sum += e[i].F;
            Union(u, v);
            // Printing the edge
            cout << u << sp << v << el;
        }
    }
}

// Prims Algorithm - Minimum Spanning Tree
// O(m log n) m - number of edges

 int spanningTree(int n, vector<vector<int>> adj[])
    {
        // code here
        // minimum priority queue
        
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        
        vector<int> vis(n, 0);
        
        // wt-node
        pq.push({0,0});
        int sum = 0;
        
        while(!pq.empty()){
            auto it = pq.top();
            pq.pop();
            int node = it.second;
            int wt = it.first;
            
            if(vis[node] ==  1) continue;
            
            // add it to mst
            vis[node] = 1; sum += wt;
            for(auto it: adj[node]){
                int adjNode = it[0], edW = it[1];
                if(!vis[adjNode]){
                    pq.push({edW, adjNode});
                    // not marking visted here
                }
            }
        }
        
        return sum;
        
    }

/*Lowest Common Algorithm
preprocessing of O(nlogn)
each LCA query in log n*/
// it can be easily done in O(n) by storing parents array


vector<vector<ll>> g(N), up(N, vector<ll> (100, -1));
vector<ll> lvl(N);

void dfs(ll node, ll par, ll level){
	lvl[node] = level;

	for(auto it: g[node]){
		if(it != par){
			dfs(it, node, level + 1);
		}
	}
}


void binary_lift(ll src, ll par){
	up[src][0] = par;
	for(ll i = 1; i < 20; i++){
		if(up[src][i-1] != -1){
			up[src][i] = up[up[src][i-1]][i-1];
		}
		else up[src][i] = -1;
	}

	for(auto it: g[src]){
		if(it != par){
			binary_lift(it, src);
		}
	}
}

ll lift_node(ll node, ll jump_req){

	for(ll i = 19; i>=0; i--){
		if(node == -1 or jump_req == 0){
			break;
		}
		if(jump_req >= (1<<i)){
			jump_req -= (1<<i);
			node = up[node][i];
		}
	}
	return node;
}

ll LCA(ll u, ll v){
	if(lvl[u] < lvl[v]){
		swap(u,v);
	}

    // u and v are now at the same level
	u = lift_node(u, lvl[u] - lvl[v]);

	if(u == v) return u;

	for(ll i = 19; i >=0; i--){
		if(up[u][i] != up[v][i]){
			u = up[u][i]; v = up[v][i];
		}
	}
	return lift_node(u,1);
}

// Dijsktras algo implementation --> Dijsktras on sparse graphs  ((n,m) --> (v,e))
// o(elogv + E)

// priority queue implementation

vector<pair<int,int>> ar[N];
void solve()
{
    ll n, m, a, b;
    cin >> n >> m;
    while (m--)
    {
        ll w;
        cin >> a >> b >> w;
        ar[a].pb({b, w});
        ar[b].pb({a, w});
    }
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq; // min - priority queue
    pq.push({0, 1});                                                                    // Pushing Distance-node pair in the priority queue

    vector<ll> dist(n + 1, INF); // distance array
    vector<ll> p(n + 1, -1);     // parent array
    // distance of 1 from source (i.e. 1) is zero
    dist[1] = 0;

    while (!pq.empty())
    {
        // current node
        ll cur = pq.top().second;
        // current node's distance from source
        ll cur_d = pq.top().first; 
        pq.pop();
        if (cur_d != dist[cur])
        {
            continue;
        }
        for (pll edge : ar[cur])
        {
            if (cur_d + edge.second < dist[edge.first])
            {
                dist[edge.first] = cur_d + edge.second;
                p[edge.first] = cur;
                pq.push({dist[edge.first], edge.first});
            }
        }
    }
    if (dist[n] == INF)
    {
        cout << -1;
        return;
    }
}

// Bellman ford algorithm - single source shortest path algorithm
// directed graph with negative edges
// maximum (n-1) iterations helps us produce the result
// negative weight cycle - if relaxation happens on nth iteration
// then we will have an negative weight cycle in the graph
            ll n,m; cin >> n >> m;
            vector<vector<ll>> g;



            for(ll i = 0; i < m; i++){
                    ll a,b,c; cin >> a >> b >> c;
                    vector<ll> temp;
                    temp.pb(a); temp.pb(b); temp.pb(c);
                    g.push_back(temp);
            }


            vector<ll> d(n+1, inf);
            vector<ll> p(n+1, -1);
            d[1] = 0;

            ll poi;
            for(int i = 0; i < n; i++){
                poi = -1;
                for(ll j = 0; j < m; j++){
                        auto it = g[j];
                        ll u = it[0], v = it[1], w = it[2];
                        if(d[u] < inf){
                            if(d[v] > d[u] + w){
                                d[v] = d[u] + w;
                                p[v] = u;
                                poi = v;
                            }
                        }

                }
            }
            // n-cycle is not present
            if(poi == -1){
                cout << "NO"; return;
            }

            // n-cycle is present
            cout << "YES" << el;

            ll y = poi;
            for (ll i = 0; i < n; ++i)
                y = p[y];

            vector<ll> path;
            for (ll cur = y;; cur = p[cur]) {
                path.push_back(cur);
                if (cur == y && path.size() > 1)
                    break;
            }
            reverse(path.begin(), path.end());

            for(auto it: path){
                cout << it << sp;
            }



// Floyd Warshall Algorithm - All source shortest path algorithm

// can have negative edges, but not negative cycles- as we have infinitely 
// small weight of distance as we can keep going through the cycle

ll n, m, q; cin >> n >> m >> q;

                vector<vector<ll>> g(n+1, vector<ll> (n+1, inf));

                for(ll i = 0; i < m; i++){
                    ll u,v; cin >> u >> v;
                    ll w; cin >> w;
                    g[u][v] = min(g[u][v], w);
                    g[v][u] = min(g[v][u], w);
                }

                for(ll i = 1; i <= n; i++){
                        g[i][i] = 0;
                }

    
                for(ll k = 1; k <= n; k++){
                    for(ll i = 1; i <= n; i++){
                        for(ll j = 1; j <= n; j++){
                            //if(g[i][k] < inf and g[k][j] < inf)
                            {
                                g[i][j] = min(g[i][j], g[i][k] + g[k][j]);
                            }
                        }
                    }
                }


                for(ll i = 0; i < q; i++){
                        ll u,v; cin >> u >> v;
                        ll ans = g[u][v];
                        if(ans == inf){
                            cout << -1 << el; continue;
                        }
                        cout << ans << el;
                }
// Kosaraju Algo

void dfs1(int node, stack<int> &st, vector<int> &vis, vector<vector<int>> &adj){
    vis[node] = 1;
    
    for(auto it: adj[node]){
        if(!vis[it]){
            dfs1(it, st, vis, adj);
        }
    }
    st.push(node);
}

void dfs2(int node, vector<int> &vis, vector<vector<int>> &g){
    vis[node] = 1;
    
    for(auto it: g[node]){
        if(!vis[it]){
            dfs2(it, vis, g);
        }
    }
}
int kosaraju(int V, vector<vector<int>>& adj)
{
    //code here
    int n = V;
    stack<int> st;
    vector<int> vis(n, 0);
    // dfs call to sort the nodes in terms of out time
    for(int i = 0; i < n; i++){
        if(vis[i] == 0){
            dfs1(i, st, vis, adj);
        }
    }
    
    vector<vector<int>> g(n);
    
    // the graph has been reversed here
    for(int j = 0; j < n; j++){
        vector<int> temp = adj[j];
        for(int i = 0; i < temp.size(); i++){
            g[temp[i]].push_back(j);
        }
    }
    
    int cnt = 0;
    for(int i = 0; i < n; i++) vis[i] = 0;
    
    // second dfs to count the cc in reversed graph
    while(!st.empty()){
        int u = st.top(); st.pop();
        if(vis[u] == 0){
            dfs2(u, vis, g);
            cnt++;
        }
    }
    
    return cnt;
    
}