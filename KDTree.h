/**
 * File: KDTree.h
 * Author Lili Meng (lilimeng1103@gmail.com) mainly based on the code by Keith Schwarz (htiek@cs.stanford.edu)
 * Thanks a lot for the discussion with Jimmy Chen, Victor Gan, Keith Schwarz, David Lowe.
 * ------------------------
 * Perform constructing trees, efficient exact query for k-nearest neighbors based on Bounded priority queue kd-tree,
 * Best-Bin-First(BBF) query for approximate k-nearest neighbors search.
 * For more BBF query, please refer to
 * Beis, J. S. and Lowe, D. G.  Shape indexing using approximate nearest-neighbor search in high-dimensional spaces.
 *
 * An interface representing a kd-tree in some number of dimensions. The tree
 * can be constructed from a set of data and then queried for membership and
 * nearest neighbors.
 **/

#ifndef KDTREE_INCLUDED
#define KDTREE_INCLUDED

#include <stdexcept>
#include <cmath>
#include <queue>

#include "Point.h"
#include "BoundedPQueue.h"
#include "ReadData.h"

using namespace std;

/// key value pair. Not use the std::pair because there is default
/// "<" operator overloading for it and it is too heavy for operation.
template <class T, class V = double>
struct KeyValue
{
    T key;
    V value;
    KeyValue(const T k, const V v) : key(k), value(v){}
};

template <size_t N, typename ElemType>
class KDTree {
public:
    /** Constructor: KDTree();
     * Usage: KDTree<3, int> myTree;
     * ----------------------------------------------------
     * Constructs an empty KDTree.
     **/
    KDTree();

    /** Destructor: ~KDTree()
     * Usage: (implicit)
     * ----------------------------------------------------
     * Cleans up all resources used by the KDTree.
     **/
    ~KDTree();

    /**
     * KDTree(const KDTree& rhs);
     * KDTree& operator=(const KDTree& rhs);
     * Usage: KDTree<3, int> one = two;
     * Usage: one = two;
     * -----------------------------------------------------
     * Deep-copies the contents of another KDTree into this one.
     **/
    KDTree(const KDTree& rhs);
    KDTree& operator=(const KDTree& rhs);

    /** size_t dimension() const;
     * Usage: size_t dim = kd.dimension();
     * ----------------------------------------------------
     * Returns the dimension of the points stored in this KDTree.
     **/
    size_t dimension() const;

    /**size_t size() const;
     * bool empty() const;
     * Usage: if (kd.empty())
     * ----------------------------------------------------
     * Returns the number of elements in the kd-tree and whether the tree is
     * empty.
     **/
    size_t size() const;
    bool empty() const;

     /** load data from file **/

    void loadData(string dataFileName, vector<vector<double>>& dataPointsVec);

    /** bool contains(const Point<N>& pt) const;
     * Usage: if (kd.contains(pt))
     * ----------------------------------------------------
     * Returns whether the specified point is contained in the KDTree.
     **/
    bool contains(const Point<N>& pt) const;

    /**void insert(const Point<N>& pt, const ElemType& value);
     * Usage: kd.insert(v, "This value is associated with v.");
     * ----------------------------------------------------
     * Inserts the point pt into the KDTree, associating it with the specified
     * value. If the element already existed in the tree, the new value will
     * overwrite the existing one.
     **/

    void insert(const Point<N>& pt, const ElemType& value);

    /** Build Tree*/
    void buildTree(vector<vector<double> > dataPointsVec, const ElemType& value);

    /** ElemType& operator[](const Point<N>& pt);
     * Usage: kd[v] = "Some Value";
     * ----------------------------------------------------
     * Returns a reference to the value associated with point pt in the KDTree.
     * If the point does not exist, then it is added to the KDTree using the
     * default value of ElemType as its key.
     **/
    ElemType& operator[](const Point<N>& pt);

    /** ElemType& at(const Point<N>& pt);
     * const ElemType& at(const Point<N>& pt) const;
     * Usage: cout << kd.at(v) << endl;
     * ----------------------------------------------------
     * Returns a reference to the key associated with the point pt. If the point
     * is not in the tree, this function throws an out_of_range exception.
     **/
    ElemType& at(const Point<N>& pt);
    const ElemType& at(const Point<N>& pt) const;

    /** ElemType kNNValue(const Point<N>& key, size_t k) const
     * Usage: cout << kd.kNNValue(v, 3) << endl;
     * ----------------------------------------------------
     * Given a point v and an integer k, finds the k points in the KDTree
     * nearest to v and returns the most common value associated with those
     * points. In the event of a tie, one of the most frequent value will be
     * chosen.
     **/
    ElemType kNNValues(const Point<N>& key, size_t k) const;

    multiset<Point<N>> getkNNPoints(const Point<N>& key, size_t k) const;

    multiset<Point<N>> getBBFKNNPoints(const Point<N>& key, size_t k, size_t maxEpoch);

private:

  /** implementation details **/

    struct TreeNode {

        Point<N> key;
        ElemType value;
        size_t level;
        //int n; number of Point<N> in one TreeNode*, that is to say, how many children does this TreeNode have?
        //int leaf;   /** 1 if node is a leaf, 0 otherwise */

        TreeNode* left;
        TreeNode* right;
    };

     // typedef to avoid ugly long declaration
    //typedef stack<TreeNode*> NodeStack;
    typedef KeyValue<TreeNode*> NodeBind;
    typedef priority_queue<NodeBind, vector<NodeBind>, greater<NodeBind> > NodeMinPQ;

    TreeNode* root;

    /** The number of elements currently stored **/
    size_t numElements;

    /** Takes in a node and recursively delete the subtree it represents
     going from its children up **/
    void deleteTreeNode(TreeNode* currentTreeNode);

    /** Helper function for copying KDTree **/
    TreeNode* copyTree(TreeNode* rootNode);

    /** KNNValue helper function of building BoundedPQueue for the KNNValue  */
    void KNNValueshelper(const Point<N>&key, BoundedPQueue<TreeNode*>& kNearestPQ, TreeNode* currentNode) const;

    /** KNNValue helper to find most common value in BoundedPriorityQueue **/
    ElemType FindMostCommonValueInPQ(BoundedPQueue<TreeNode*> nearestPQ) const;

    /** function for traversing to the leaf＊*/
    TreeNode* exploreToLeaf(Point<N> pt, TreeNode* root, NodeMinPQ& container);

    /** function for counting the number of Point<N> under one TreeNode*/
    int countNodes(TreeNode* root);  // that is to say, how many children nodes does this TreeNode have?

};

/** KDTree class implementation details */

/** Constructor **/
template <size_t N, typename ElemType>
KDTree<N, ElemType>::KDTree() {
    numElements = 0;
    root = NULL;
}

/** Destructor **/
template <size_t N, typename ElemType>
KDTree<N, ElemType>::~KDTree() {
    deleteTreeNode(root);
    numElements = 0;
}

/** Helper function of deleting the current TreeNode**/
template <size_t N, typename ElemType>
void KDTree<N, ElemType>::deleteTreeNode(TreeNode* currentNode){

    if(currentNode == NULL) return;
    /**Recursion**/
    deleteTreeNode(currentNode->left);
    deleteTreeNode(currentNode->right);
    delete currentNode;
}

/**Copy Constructor **/
template <size_t N, typename ElemType>
KDTree<N, ElemType>::KDTree(const KDTree& rhs) {

    root = copyTree(rhs.root);
    numElements = rhs.numElements;
}

/** Assignment operator, clears old tree if not the same tree and copies
 *  the 'other' tree into the new tree
 */
template <size_t N, typename ElemType>
KDTree<N, ElemType>& KDTree<N, ElemType>::operator=(const KDTree& rhs){

    if (this !=&rhs)
    {
        deleteTreeNode(this->root);
        root=copyTree(rhs.root);
        numElements = rhs.numElements;
    }

    return *this;
}

/** CopyTree **/
template <size_t N, typename ElemType>
typename KDTree<N, ElemType>::TreeNode* KDTree<N, ElemType>::copyTree(TreeNode* rootNode){

    if(rootNode==NULL) return NULL;

    TreeNode* rootNodeCopy = new TreeNode;

    rootNodeCopy->key = rootNode->key;
    rootNodeCopy->value = rootNode->value;
    rootNodeCopy->level = rootNode->level;

    /** Recursion**/
    rootNodeCopy->left = copyTree(rootNode->left);
    rootNodeCopy->right = copyTree(rootNode->right);

    return rootNodeCopy;
}


template <size_t N, typename ElemType>
size_t KDTree<N, ElemType>::dimension() const {

    return N;
}

/** Return the number of elements currently stored **/
template <size_t N, typename ElemType>
size_t KDTree<N, ElemType>::size() const{

    return numElements;
}

/** Returns whether the it's empty**/
template <size_t N, typename ElemType>
bool KDTree<N, ElemType>::empty() const{

    if(numElements==0)
    return true;
    else
    return false;
}

/**
 * Returns whether the specified Point is contained in the KDTree
 **/
template <size_t N, typename ElemType>
bool KDTree<N, ElemType>::contains(const Point<N>& pt) const{

    TreeNode* currentNode = root;
    while(currentNode!=NULL)
    {
        if(currentNode->key==pt)
        {
            return true;
        }

        /** compares the parts of the pt to determine which subtree to look into next, image N=2, that is binary search tree **/
        if(pt[currentNode->level % N] >= currentNode->key[currentNode->level %N])
        {
            currentNode = currentNode->right;
        }
        else
        {
            currentNode = currentNode->left;
        }

    }

    return false;
}

template <size_t N, typename ElemType>
void KDTree<N, ElemType>::loadData(string dataFileName, vector<vector<double>>& dataPointsVec) {

    ReadData rd("sample_data.txt");

    dataPointsVec=rd.allDataPointsVec;
}


/** Insert the specified point into the KDTree with associated value. If the point already exists in the KDTree, the old value is overwritten **/
template <size_t N, typename ElemType>
void KDTree<N, ElemType>::insert(const Point<N>& pt, const ElemType& value) {


    TreeNode* currentNode = root;
    TreeNode* prevNode = NULL;
    size_t level = 0;

    while(currentNode!=NULL)
    {
        ++level;
        if(pt==currentNode->key)
        {
            currentNode->value = value;
        }

        size_t dim = currentNode->level % N;

        // If pt[dim] >= currentNode->key[dim],insert the node to the right node.

        if(pt[dim] < currentNode->key[dim]){
            prevNode = currentNode;
            currentNode = currentNode->left;
        }
        else {
            prevNode = currentNode;
            currentNode = currentNode->right;
        }
    }


    ++numElements;

    TreeNode* newNode = new TreeNode;
    newNode->key = pt;
    newNode->value = value;
    newNode->left = NULL;
    newNode->right = NULL;

    if(currentNode == root){
        root = newNode;
    }
    else {
        if(pt[prevNode->level % N]<prevNode->key[prevNode->level % N])
        {
            prevNode->left = newNode;
        }
        else
        {
            prevNode->right = newNode;
        }
    }
}

template <size_t N, typename ElemType>
void KDTree<N, ElemType>::buildTree(vector<vector<double> > dataPointsVec, const ElemType& value)
{

    Point<N> dataPoint;
    for(int i=0; i<dataPointsVec.size(); i++)
    {
        for(int j=0; j<dataPointsVec[i].size(); j++)
        {
            dataPoint.coords[j]=dataPointsVec[i][j];
        }

    }

     for(int i=0; i<dataPointsVec.size(); i++)
    {

        insert(dataPoint, value);

    }
}


/** Returns a reference to the value associated with the point pt. If the point does not exist in the KDTree, it is added with
 * the default value of ElemType as its value, and a reference to this new value is returned. This is the same behavior as the
 * STL map's operator[]
 * Note that this function does not have a const overload because the function may mutate the tree
 */
template <size_t N, typename ElemType>
ElemType& KDTree<N, ElemType>::operator[](const Point<N>& pt){

    TreeNode* currentNode = root;
    TreeNode* prevNode = NULL;
    size_t level = 0;
    while (currentNode != NULL){
        ++level;

        if(pt==currentNode->key){
            return currentNode->value;
        }

        size_t dim = currentNode->level % N;

        if(pt[dim] < currentNode->key[dim]) {
            prevNode = currentNode;
            currentNode = currentNode->left;
        }
        else {
            prevNode = currentNode;
            currentNode = currentNode->right;
        }
    }

    ++numElements;

    //Make the new node to insert into the KDTree
    TreeNode* newNode = new TreeNode;
    newNode->key = pt;
    newNode->value = ElemType();
    newNode->level = level;
    newNode->left = NULL;
    newNode->right = NULL;

    if(currentNode == root) {
        root = newNode;
        return newNode->value;
    }
    else {
        if(pt[prevNode->level % N] >= prevNode->key[prevNode->level % N])
        {
            prevNode->right = newNode;
        }
        else
        {
            prevNode->left = newNode;
        }

        return newNode->value;
    }

}

/** Returns a reference to the value associated with the point pt, if it exists. If the point is not in the tree,
 * then this function throws an out_of_range exception
 **/
template <size_t N, typename ElemType>
ElemType& KDTree<N, ElemType>::at(const Point<N>& pt){

    TreeNode* currentNode = root;
    while (currentNode != NULL) {
        if(currentNode->key == pt) {
            return currentNode->value;
        }

        //Find the dim
        size_t dim = currentNode->level % N;

        //compare the approximate indices
        if(pt[dim] < currentNode->key[dim]) {
            currentNode = currentNode->left;
        }
        else {
            currentNode = currentNode->right;
        }
    }

    throw out_of_range("That point does not exist");
}

/** This function is const-overloaded, since it does not change the tree **/
template <size_t N, typename ElemType>
const ElemType& KDTree<N, ElemType>::at(const Point<N>& pt) const {

     TreeNode* currentNode = root;
     while(currentNode != NULL) {
        if(currentNode->key == pt) {
            return currentNode->value;
        }

        size_t dim = currentNode->level % N;

        if(pt[dim] < currentNode ->key[dim]) {
            currentNode = currentNode->left;
        }
        else {
            currentNode = currentNode->right;
        }
     }

    throw out_of_range("That point does not exist");

}


/** Exact Query of K Nearest-Neigbour **/
template <size_t N, typename ElemType>
ElemType KDTree<N, ElemType>::kNNValues(const Point<N>& key, size_t k) const {

    BoundedPQueue<TreeNode*> kNearestPQ(k);
    KNNValueshelper(key, kNearestPQ, root);

    return FindMostCommonValueInPQ(kNearestPQ);
}

/** Exact Query of K Nearest-Neigbour **/
template <size_t N, typename ElemType>
multiset<Point<N>> KDTree<N, ElemType>::getkNNPoints(const Point<N>& key, size_t k) const {

    BoundedPQueue<TreeNode*> kNearestPQ(k);
    KNNValueshelper(key, kNearestPQ, root);

    multiset<Point<N>> kNNPoints;
    while(!kNearestPQ.empty())
    {
        kNNPoints.insert((kNearestPQ.dequeueMin())->key);
    }
    return kNNPoints;
}

/**
 * KNNValueshelper(key, bpq, currentNode)
 * key--query point
 * A helper function of building the bounded priority queue of points nearest to the query point in the KDTree
 **/

template<size_t N, typename ElemType>
void KDTree<N, ElemType>::KNNValueshelper(const Point<N>& key, BoundedPQueue<TreeNode*>& kNearestPQ, TreeNode* currentNode) const {

    if (currentNode == NULL) return;

    kNearestPQ.enqueue(currentNode, Distance(currentNode->key, key));

    size_t dim = currentNode->level % N;

    //like Binary Search Tree, if key[dim] < currentNode->key[dim], turn to the left of the tree
    if(key[dim] < currentNode->key[dim])
    {
        KNNValueshelper(key, kNearestPQ, currentNode->left);

        // If the query hypersphere crosses the splitting plane, check the other subtree
        if(kNearestPQ.size() < kNearestPQ.maxSize() || fabs(currentNode->key[dim] - key[dim]) < kNearestPQ.worst() ) {

            KNNValueshelper(key, kNearestPQ, currentNode->right);
        }
    }
    else //like Binary Search Tree, if key[dim] >= currentNode->key[dim], turn to the right of the tree
    {
        KNNValueshelper(key, kNearestPQ, currentNode->right);

        // If the hypersphere crosses the splitting plane, check the other subtree
        if(kNearestPQ.size() < kNearestPQ.maxSize() || fabs(currentNode->key[dim] - key[dim]) < kNearestPQ.worst() ) {

            KNNValueshelper(key, kNearestPQ, currentNode->left);
        }

    }
}

/*
 * FindMostCommonValueInPQ(bpq)
 * Takes in a bounded priority queue of Node*'s in the KDTree and
 * returns the most common value stored in the nodes.
 */
template<size_t N, typename ElemType>
ElemType KDTree<N, ElemType>::FindMostCommonValueInPQ(BoundedPQueue<TreeNode*> nearestPQ) const{
    multiset<ElemType> values;
    while(!nearestPQ.empty()) {
        values.insert((nearestPQ.dequeueMin())->value);
    }

    ElemType best;
    size_t bestFrequency = 0;
    for(typename multiset<ElemType>::iterator it = values.begin(); it !=values.end(); ++it) {
        if (values.count(*it) > bestFrequency) {
            best = *it;
            bestFrequency = values.count(*it);
        }
    }
    return best;
}

/** Function for traversing the tree **/
template <size_t N, typename ElemType>
typename KDTree<N, ElemType>::TreeNode* KDTree<N, ElemType>::exploreToLeaf(Point<N> pt, TreeNode* root, NodeMinPQ& minPQ)
{
    TreeNode* untraversed;   // untraversed storing the untraversed TreeNode* on the KD Tree
    TreeNode* currentNode = root;

    double value;
    size_t dim;

    while(currentNode!=NULL && currentNode->left!=NULL && currentNode->right!=NULL)   //currentNode->left!=NULL && currentNode->right!=NULL signifies that currentNode is not a leaf
    {
        //partition dimension and value;
        dim = currentNode->level % N;
        value = currentNode->value;

        if(dim>=N)
        {
            cout<<"error, comparing incompatible points"<<endl;
            return NULL;
        }

        // go to a child and preserve the other
        if(pt[dim] < value)
        {
            untraversed = currentNode->right;
            currentNode = currentNode->left;
        }
        else
        {
            untraversed = currentNode->left;
            currentNode = currentNode->right;
        }

        if (untraversed!=NULL)
        {
            minPQ.push(NodeBind(untraversed, fabs(untraversed->value- pt[untraversed->level % N])));
        }

    }

    return currentNode;
}

/**
    * Search for approximate k nearest neighbours using Best-Bin-First (BBF) approach
    * @param key        Query point data
    * @param k          number of nearest neighbour returned
    * @param maxEpoch   maximum search epoch
    **/
template <size_t N, typename ElemType>
multiset<Point<N>> KDTree<N, ElemType>::getBBFKNNPoints(const Point<N>& key, size_t k, size_t maxEpoch) {

    multiset<Point<N>> result;

    size_t epoch = 0;

    TreeNode* currentNode = root;

    NodeMinPQ minPQ;

    BoundedPQueue<TreeNode*> kNearestPQ(k);

    double currentBest = numeric_limits<double>::max();

    double dist = 0;

    minPQ.push(NodeBind(this->root, 0));

    while(!minPQ.empty() && epoch < maxEpoch)
    {
        currentNode = minPQ.top().key;
        minPQ.pop();

        //find leaf node and push unexplored to minPQ
        currentNode = exploreToLeaf(key, currentNode, minPQ);
    }

    dist = Distance(currentNode->key, key);

    if(dist < currentBest)
    {
        if(kNearestPQ.size()==k)
        {
            //update the currentBest;
            kNearestPQ.pop();
            kNearestPQ.enqueue(currentNode, Distance(currentNode->key, key));
            currentBest = kNearestPQ.best();
        }
        else if(kNearestPQ.size() == k-1)
        {
            kNearestPQ.enqueue(currentNode, Distance(currentNode->key, key));
            currentBest = kNearestPQ.best();
        }
        else
        {
            kNearestPQ.enqueue(currentNode, Distance(currentNode->key, key));
        }
    }

    ++epoch;

    while(!kNearestPQ.empty())
    {
        result.insert((kNearestPQ.dequeueMin())->key);
    }

    return result;
}

#endif // KDTREE_INCLUDED
