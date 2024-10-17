const fs = require('fs');
const { PCA } = require('ml-pca');

const DISTANCE_SCALE_PX = 1000;

// Read input files
const clusteredTweets = JSON.parse(fs.readFileSync('clustered_tweets.json', 'utf8'));
const tweetsWithEmbeddings = JSON.parse(fs.readFileSync('tweets_with_embeddings.json', 'utf8')).slice(0, 3000);

// Extract embeddings from tweets and centroids
const tweetEmbeddings = tweetsWithEmbeddings.map(tweet => tweet.embedding);
const centroidEmbeddings = clusteredTweets.map(cluster => cluster.centroid);
const allEmbeddings = [...tweetEmbeddings];


// Perform PCA
const pca = new PCA(allEmbeddings);
console.log('Constructed PCA model');
const { data: pcaResults } = pca.predict(allEmbeddings, { nComponents: 2 });
console.log('Predicted 10 results');

// Separate PCA results for centroids and tweets
const tweetPositions = pcaResults;

// Scale positions to -950 to 950 range (leaving some margin)
function scalePositions(positions) {
  const xValues = positions.map(p => p[0]);
  const yValues = positions.map(p => p[1]);
  const maxAbsX = Math.max(Math.abs(Math.min(...xValues)), Math.abs(Math.max(...xValues)));
  const maxAbsY = Math.max(Math.abs(Math.min(...yValues)), Math.abs(Math.max(...yValues)));
  const maxAbs = Math.max(maxAbsX, maxAbsY);

  return positions.map(p => [
    (p[0] / maxAbs) * DISTANCE_SCALE_PX,
    (p[1] / maxAbs) * DISTANCE_SCALE_PX
  ]);
}


const scaledTweetPositions = scalePositions(tweetPositions);

// Calculate Euclidean distance
function euclideanDistance(vec1, vec2) {
  return Math.sqrt(vec1.reduce((sum, val, i) => sum + Math.pow(val - vec2[i], 2), 0));
}

// Find closest cluster
function findClosestCluster(tweetEmbedding, centroids) {
  let closestCluster = 0;
  let minDistance = Infinity;
  centroids.forEach((centroid, index) => {
    const distance = euclideanDistance(tweetEmbedding, centroid);
    if (distance < minDistance) {
      minDistance = distance;
      closestCluster = index;
    }
  });
  return closestCluster;
}


const tweetIdToIndex = new Map();


// Create tweet nodes
const tweetNodes = tweetsWithEmbeddings.map((tweet, index) => {
  const closestCluster = findClosestCluster(tweet.embedding, centroidEmbeddings);
  tweetIdToIndex.set(tweet.tweet.id_str, index);

  return {
    key: `tweet_${tweet.tweet.id_str}`,
    label: tweet.tweet.full_text.substring(0, 50) + (tweet.tweet.full_text.length > 50 ? '...' : ''),
    tag: 'tweet',
    URL: `https://twitter.com/i/web/status/${tweet.tweet.id_str}`,
    cluster: closestCluster.toString(),
    x: scaledTweetPositions[index][0],
    y: scaledTweetPositions[index][1],
    score: Math.max(Math.min(tweet.tweet.favorite_count + tweet.tweet.retweet_count, 10), 1) / 10,
  };
});

// Combine all nodes
const allNodes = tweetNodes;

const edges = [];
tweetsWithEmbeddings.forEach(tweet => {
  if (tweet.tweet.in_reply_to_status_id_str) {
    const sourceIndex = tweetIdToIndex.get(tweet.tweet.id_str);
    const targetIndex = tweetIdToIndex.get(tweet.tweet.in_reply_to_status_id_str);
    if (sourceIndex !== undefined && targetIndex !== undefined) {
      edges.push([`tweet_${tweet.tweet.id_str}`, `tweet_${tweet.tweet.in_reply_to_status_id_str}`]);
    }
  }
});

// Create graph object
const graph = {
  nodes: allNodes,
  edges,
  clusters: clusteredTweets.map((cluster, index) => ({
    key: index.toString(),
    color: `#${Math.floor(Math.random()*16777215).toString(16)}`,
    clusterLabel: cluster.name,
    // x: scaledCentroidPositions[index][0],
    // y: scaledCentroidPositions[index][1]
  })),
  tags: [
    { key: "centroid", image: "centroid.svg" },
    { key: "tweet", image: "tweet.svg" }
  ]
};

// Write output file
fs.writeFileSync("packages/demo/public/graph.json", JSON.stringify(graph, null, 2));

console.log('Graph data has been written to packages/demo/public/graph.json');