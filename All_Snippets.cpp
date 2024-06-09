/* Snippets to learn 
1. Euclidean Algo/ Extended Euclidean Algo
2. KMP (Prefix Function) 
3. Z - Array
4. Manacher
5. Rolling Hash
6. Own unordered map/ Hash Table
7. Minimum Queue/ minimum stack
8. Implement Heap
8. Policy Based Data Structure */

// KMP Prefix Function
/*prefix[i] is defined as the length of the longest proper prefix 
of the substring s[0..i], which is also the suffix of the substring
by definition : prefix[0] = 0
prefix[i] : max. k such that s[0..k-1] = s[i-(k-1)..i]*/

/*
Applications
1. Search for a substring in the string
2. Counting the number of occurences of each prefix
3. Number of different substring in a string
4. Compressing a string */


vector<int> prefix_function(string s) {
    int n = (int)s.length();
    vector<int> pi(n, 0);
    for (int i = 1; i < n; i++) {
        int j = pi[i-1];
        while (j > 0 && s[i] != s[j])
            j = pi[j-1];
        if (s[i] == s[j])
            j++;
        pi[i] = j;
    }
    return pi;
}

// fast unordered map use to avoid collisions
struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {
        // http://xorshift.di.unimi.it/splitmix64.c
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }
    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }
};


// Disjoint Sets Union
class DSU{
public:
	int parent[N+1];
	int Rank[N+1];
	DSU(int n)
	{
		for(int i=1;i<=n;i++)
		{
			parent[i]=i;
			Rank[i]=0;
		}
	}
	// O(alpha(n)) - alpha(n) is the inverse ackerman function 
	int findPar(int node)
	{
		if(node == parent[node])
			return node;
		else 
			return parent[node] = findPar(parent[node]);
	}
	void Union(int u, int v)
	{
		u=findPar(u);
		v=findPar(v);
		if(u!=v)
		{
			if(Rank[u]  < Rank[v])
			{
				parent[u]=v;
			}	
			else if(Rank[u]>Rank[v])
			{
				parent[v]=u;
			}
			else
			{
				parent[v]=u;
				Rank[u]++;
			}
		}
	}	
};   

// Binary Modular Exponentiation
long long binpow(long long a, long long b, long long m) {
    a %= m;
    long long res = 1;
    while (b > 0) {
        if (b & 1)
            res = res * a % m;
        a = a * a % m;
        b >>= 1;
    }
    return res;
}

// multiply two numbers without using multiplication operator
int multiplyTwoNumbers(int a, int b) {
   int result = 0;
   while (b > 0) {
      if (b & 1) {
         result += a;
         }
      a = a << 1;
      b = b >> 1;
   }
   return result;
}

// AND for range [L,R]
ll andy(ll m, ll n) {
    ll i = 0;
    while(m != n){
        m >>= 1LL;
        n >>= 1LL;
        i++;
    }
    return n << i;
}

// sieve of eratothenes
vector<ll> prim(N, 1);
void pcalc(){
		prim[0] = prim[1] = 0;
		for(ll i = 2; i * i <= N; i++){
				if(prim[i] == 1){
						for(ll j = i*i; j <= N; j += i){
							prim[j] = 0;
						}
				}
		}
}

// smallest prime factor calculation using sieve
void leastPrimeFactor(int n)
{
    // Create a vector to store least primes.
    // Initialize all entries as 0.
    vector<int> least_prime(n+1, 0);
 
    // We need to print 1 for 1.
    least_prime[1] = 1;
 
    for (int i = 2; i <= n; i++)
    {
        // least_prime[i] == 0
        // means it i is prime
        if (least_prime[i] == 0)
        {
            // marking the prime number
            // as its own lpf
            least_prime[i] = i;
 
            // mark it as a divisor for all its
            // multiples if not already marked
            for (int j = i*i; j <= n; j += i)
                if (least_prime[j] == 0)
                   least_prime[j] = i;
        }
    }
 
    // print least prime factor of
    // of numbers till n
    for (int i = 1; i <= n; i++)
        cout << "Least Prime factor of "
             << i << ": " << least_prime[i] << "\n";
}

// prime factorisation after 
// finding least prime factor
void LPF(){
    spf[1] = 1;

    for(int i = 2; i <= N; i++){
        spf[i] = i;
    }

    for(int i = 2; i <= N; i++){
        if(spf[i] == i){ // this means i is prime
            for(int j = 2*i; j <= N; j+=i){
                if(spf[j] == j){
                    spf[j] = i;
                }
            }
        }
    }

    int x; // we need to find prime factorisation of x

    vector<int> ans;
    while(x){
        ans.push_back(spf[x]);
        x /= spf[x];
    }

    return ans;
}

// best ncr code
vector <ll> fact(100005,1);
void facting(ll mod){
	for(ll i=1;i<=100001;i++)
	{
		fact[i]=(((i%mod)*(fact[i-1]%mod))%mod);
	}
}

// Best Factorial Code
ll PowI(ll a,ll b,ll m)
{ll ans=1%m;while(b>0){if(b%2) ans=(((ans%m)*(a%m))  %m);
a=(((a%m)*(a%m))%m); b=(ll)(b/((ll)2));}return ans;}

//nCr Function
ll nCr(ll n,ll r,ll mod){
	if(r>n)
	return -1;
	ll ri=PowI(fact[r],mod-2,mod);
	ll nri=PowI(fact[n-r],mod-2,mod);
	ll ans=(((fact[n]%mod)*(ri%mod))%mod);
	ans=(((ans%mod)*(nri%mod))%mod);
	ans%=mod;
	ans=(ans+mod)%mod;
	return ans;
}

// segment tree snippet

class SGTree{
            vector<int> seg;
        public:
            SGTree(int n){
                seg.resize(4*n+1);
            }
            // builds the segment tree
            void build(int ind, int low, int high, vector<int> &arr){
                        if(low == high){
                            seg[ind] = arr[low];
                            return;
                        }

                        int mid = (low+high)/2;

                        build(2*ind+1, low, mid, arr);
                        build(2*ind+2, mid+1, high, arr);

                        seg[ind] = min(seg[2*ind+1], seg[2*ind+2]);
            }

            // gives minimum in a segment of the tree
            int query(int ind, int low, int high, int l, int r){
                        // no overlap
                        // l r low high or low high l r
                        if(low > r or l > high){
                            return INT_MAX;
                        }

                        // complete overlap
                        // l low high r
                        if(low >= l and r >= high){
                            return seg[ind];
                        }
                        
                        // partial overlap

                        int mid = (low + high) >> 1;
                        int left = query(2*ind+1, low, mid, l,r);
                        int right = query(2*ind+2, mid+1, high, l, r);

                        return min(left,right);   
            }

            // update a value in the array
            void update(int ind, int low, int high, int i, int val){

                if(low == high){
                    seg[ind] = val; return;
                }

                int mid = (low + high) >> 1;

                if(i <= mid){
                        update(2*ind+1, low, mid, i, val);
                }
                else{
                        update(2*ind+2, mid+1, high, i, val);
                }

                seg[ind] = min(seg[2*ind+1], seg[2*ind+2]);
            }
};

// lazy propagation - range updates
// rules, if given a range [l..r] -> +val
// traverse in segment tree
class STLazy {
	vector<int> seg, lazy; 
public: 
	STLazy(int n) {
		seg.resize(4 * n); 
		lazy.resize(4 * n); 
	}
public: 
	void build(int ind, int low, int high, vector<int> &arr) {
		if(low == high) {
			seg[ind] = arr[low];
			return; 
		}
		int mid = (low + high) >> 1; 
		build(2*ind+1, low, mid, arr); 
		build(2*ind+2, mid+1, high, arr); 
		seg[ind] = seg[2*ind+1] + seg[2*ind+2];
	}
public:
	void update(int ind, int low, int high, int l, int r, 
		int val) {
		// update the previous remaining updates 
		// and propogate downwards 
		if(lazy[ind] != 0) {
			seg[ind] += (high - low + 1) * lazy[ind]; 
			// propogate the lazy update downwards
			// for the remaining nodes to get updated 
			if(low != high) {
				lazy[2*ind+1] += lazy[ind]; 
				lazy[2*ind+2] += lazy[ind]; 
			}
 
			lazy[ind] = 0; 
		}
 
		// no overlap 
		// we don't do anything and return 
		// low high l r or l r low high 
		if(high < l or r < low) {
			return; 
		}
 
		// complete overlap 
		// l low high r 
		if(low>=l && high <= r) {
			seg[ind] += (high - low + 1) * val; 
			// if a leaf node, it will have childrens
			if(low != high) {
				lazy[2*ind+1] += val; 
				lazy[2*ind+2] += val; 
			}
			return; 
		}
		// last case has to be no overlap case
		int mid = (low + high) >> 1; 
		update(2*ind+1, low, mid, l, r, val);
		update(2*ind+2, mid+1, high, l, r, val); 
		seg[ind] = seg[2*ind+1] + seg[2*ind+2]; 
	}
public: 
	int query(int ind, int low, int high, int l, int r) {
 
		// update if any updates are remaining 
		// as the node will stay fresh and updated 
		if(lazy[ind] != 0) {
			seg[ind] += (high - low + 1) * lazy[ind]; 
			// propogate the lazy update downwards
			// for the remaining nodes to get updated 
			if(low != high) {
				lazy[2*ind+1] += lazy[ind]; 
				lazy[2*ind+2] += lazy[ind]; 
			}
 
			lazy[ind] = 0; 
		}
 
		// no overlap return 0; 
		if(high < l or r < low) {
			return 0; 
		}
 
		// complete overlap 
		if(low>=l && high <= r) return seg[ind]; 
 
		int mid = (low + high) >> 1; 
		int left = query(2*ind+1, low, mid, l, r);
		int right = query(2*ind+2, mid+1, high, l, r);
		return left + right; 
	}
};

// Fenwick Tree
struct FenwickTree{
     vector<ll> bit;
     int n;

     FenwickTree(int nn){
          n=nn;
          bit=vector<ll>(n,0);
     }

     ll sum(int r){
        ll ret=0;
        while(r>=0){
            ret+=bit[r];
            r=(r&(r+1))-1;
        }
        return ret;
     }

     ll sum(int l,int r){
        return sum(r)-sum(l-1);
     }

     void add(int idx,ll delta){
         while(idx<n){
             bit[idx]+=delta;
             idx=(idx|idx+1);
         }
     }

     int find_kth(ll x){
          int pos=0;
          for(int i=20;i>=0;i--){
              if((pos+(1<<i)-1)<n && x>bit[pos+(1<<i)-1]){
                   x-=bit[pos+(1<<i)-1];
                   pos+=(1<<i);
              }
          }

          return pos;
     }
};
