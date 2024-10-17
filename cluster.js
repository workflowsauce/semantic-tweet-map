const fs = require('fs');
const math = require('mathjs');

// Load the tweets
const tweets = JSON.parse(fs.readFileSync('tweets_with_embeddings.json', 'utf8'));

// Extract embeddings
const embeddings = tweets.map(t => t.embedding);

// K-means clustering function
function kMeans(data, k, maxIterations = 100) {
  console.log('Clustering data');
    // Initialize centroids randomly
    let centroids = data.slice(0, k);
    
    for (let iter = 0; iter < maxIterations; iter++) {
        // Assign points to clusters
        const clusters = Array(k).fill().map(() => []);
        for (let i = 0; i < data.length; i++) {
            const distances = centroids.map(c => math.distance(data[i], c));
            const closestCentroidIndex = distances.indexOf(Math.min(...distances));
            clusters[closestCentroidIndex].push(i);
        }
        
        // Update centroids
        const newCentroids = clusters.map(cluster => {
            if (cluster.length === 0) return centroids[clusters.indexOf(cluster)];
            return math.mean(cluster.map(i => data[i]), 0);
        });
        
        // Check for convergence
        if (math.deepEqual(newCentroids, centroids)) break;

        // Calculate distance/difference between newCentroids and centroids
        const centroidDifference = math.mean(newCentroids.map((newCentroid, index) => 
            math.distance(newCentroid, centroids[index])
        ));

        // Calculate the percentage change
        const averageCentroidMagnitude = math.mean(centroids.map(c => math.norm(c)));
        const percentageChange = (centroidDifference / averageCentroidMagnitude) * 100;

        console.log(`Centroid difference: ${centroidDifference.toFixed(6)} (${percentageChange.toFixed(2)}%)`);

        // If the difference is very small, we can consider it converged
        if (percentageChange < 0.01) {
            console.log(`Converged after ${iter + 1} iterations`);
            break;
        }
        centroids = newCentroids;

    }
    
    return centroids;
}

// Cosine similarity function
function cosineSimilarity(a, b) {
    return math.dot(a, b) / (math.norm(a) * math.norm(b));
}

// Number of clusters
const k = 5;

// Perform clustering
const centroids = kMeans(embeddings, k);

// Find closest tweets to centroids
const result = centroids.map(centroid => {
    const similarities = embeddings.map(e => cosineSimilarity(centroid, e));
    const sortedIndices = similarities.map((s, i) => [s, i])
        .sort((a, b) => b[0] - a[0])
        .slice(0, 10)
        .map(([_, i]) => i);
    
    return {
        centroid: centroid,
        tweets: sortedIndices.map(i => tweets[i].tweet.full_text)
    };
});

// Write result to file
fs.writeFileSync('clustered_tweets.json', JSON.stringify(result, null, 2));

console.log('Clustering complete. Results written to clustered_tweets.json');