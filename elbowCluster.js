require('dotenv-safe').config();
const fs = require("fs");
const math = require("mathjs");
const axios = require("axios");


const NEAREST_TWEETS_SAMPLE_SIZE = 25;
const MIN_CLUSTER_SIZE = 10;
const MAX_CLUSTER_SIZE = 50;
const CLUSTER_ITERATION = 5;


async function getClusterNameFromChatGPT(tweets) {
  const API_KEY = process.env.OPENAI_API_KEY; // Make sure to set this environment variable
  const API_URL = "https://api.openai.com/v1/chat/completions";

  const prompt = `Given the following tweets, suggest a short, descriptive name for the cluster they represent. The name should be 2-5 words long. Tweets:\n\n${tweets.join("\n\n")}`;

  try {
    const response = await axios.post(
      API_URL,
      {
        model: "gpt-4o",
        messages: [{ role: "user", content: prompt }],
        temperature: 0.7,
        max_tokens: 50,
      },
      {
        headers: {
          Authorization: `Bearer ${API_KEY}`,
          "Content-Type": "application/json",
        },
      },
    );

    return response.data.choices[0].message.content.trim();
  } catch (error) {
    console.error("Error getting cluster name from ChatGPT:", error);
    return "Unnamed Cluster";
  }
}

// Load the tweets
const tweets = JSON.parse(fs.readFileSync("tweets_with_embeddings.json", "utf8"));

// Extract embeddings
const embeddings = tweets.map((t) => t.embedding);

// K-means clustering function
function kMeans(data, k, maxIterations = 100) {
  console.log(`Clustering data with ${k} clusters...`);
  // Initialize centroids randomly
  let centroids = data.slice(0, k);

  for (let iter = 0; iter < maxIterations; iter++) {
    // Assign points to clusters
    const clusters = Array(k)
      .fill()
      .map(() => []);
    for (let i = 0; i < data.length; i++) {
      const distances = centroids.map((c) => math.distance(data[i], c));
      const closestCentroidIndex = distances.indexOf(Math.min(...distances));
      clusters[closestCentroidIndex].push(i);
    }

    // Update centroids
    const newCentroids = clusters.map((cluster) => {
      if (cluster.length === 0) return centroids[clusters.indexOf(cluster)];
      return math.mean(
        cluster.map((i) => data[i]),
        0,
      );
    });

    // Check for convergence
    if (math.deepEqual(newCentroids, centroids)) break;

    // Calculate distance/difference between newCentroids and centroids
    const centroidDifference = math.mean(
      newCentroids.map((newCentroid, index) => math.distance(newCentroid, centroids[index])),
    );

    // Calculate the percentage change
    const averageCentroidMagnitude = math.mean(centroids.map((c) => math.norm(c)));
    const percentageChange = (centroidDifference / averageCentroidMagnitude) * 100;

    console.log(
      `${k} clusters iteration #${iter} centroid difference: ${centroidDifference.toFixed(6)} (${percentageChange.toFixed(2)}%)`,
    );

    // If the difference is very small, we can consider it converged
    if (percentageChange < 0.01) {
      console.log(`Converged after ${iter + 1} iterations`);
      break;
    }
    centroids = newCentroids;
  }

  return centroids;
}

// Calculate the sum of squared distances of samples to their closest cluster center
function calculateWCSS(data, centroids) {
  return data.reduce((sum, point) => {
    const distances = centroids.map((c) => math.distance(point, c));
    const minDistance = Math.min(...distances);
    return sum + minDistance ** 2;
  }, 0);
}

// Cosine similarity function
function cosineSimilarity(a, b) {
  return math.dot(a, b) / (math.norm(a) * math.norm(b));
}

// Find closest tweets to centroids
async function findClosestTweets(centroids, embeddings, tweets) {
  return await Promise.all(
    centroids.map(async (centroid) => {
      const similarities = embeddings.map((e) => cosineSimilarity(centroid, e));
      const sortedIndices = similarities
        .map((s, i) => [s, i])
        .sort((a, b) => b[0] - a[0])
        .slice(0, NEAREST_TWEETS_SAMPLE_SIZE)
        .map(([_, i]) => i);

      const closestTweets = sortedIndices.map((i) => tweets[i].tweet.full_text);
      const clusterName = await getClusterNameFromChatGPT(closestTweets);

      return {
        centroid: centroid,
        tweets: closestTweets,
        name: clusterName,
      };
    }),
  );
}

// Elbow method to find optimal k
async function findOptimalK(data, maxK) {
  console.log("Finding optimal number of clusters...");
  let bestWCSS = Infinity;
  let bestK = MIN_CLUSTER_SIZE;
  let bestCentroids = null;

  for (let k = MIN_CLUSTER_SIZE; k <= maxK; k += CLUSTER_ITERATION) {
    console.log(`Testing k=${k}...`);
    const centroids = kMeans(data, k);
    const wcss = calculateWCSS(data, centroids);

    console.log(`WCSS for k=${k}: ${wcss}`);

    if (wcss < bestWCSS) {
      bestWCSS = wcss;
      bestK = k;
      bestCentroids = centroids;

      // Save the best result so far
      const result = await findClosestTweets(bestCentroids, embeddings, tweets);

      fs.writeFileSync(`best_clusters_1_through_${k}_k${bestK}.json`, JSON.stringify(result, null, 2));
      console.log(`Updated best clusters (k=${bestK}) saved to best_clusters_1_through_${k}_k${bestK}.json`);
    } else {
      // If WCSS starts increasing, we've found the elbow
      console.log(`Elbow found at k=${bestK}`);
      break;
    }
  }

  console.log("Optimal K:", bestK);
  return { k: bestK, centroids: bestCentroids };
}

// Find optimal number of clusters
const maxK = Math.min(MAX_CLUSTER_SIZE, Math.floor(Math.sqrt(embeddings.length)));
const { k, centroids } = findOptimalK(embeddings, maxK);

console.log(`Final clustering with k=${k}...`);

// The best result is already saved, so we don't need to do anything else here

console.log(`Clustering complete. Final results written to best_clusters_k${k}.json`);
