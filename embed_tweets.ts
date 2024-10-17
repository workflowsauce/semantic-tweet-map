import dotenv from "dotenv-safe";
import fs from "fs/promises";
import OpenAI from "openai";
import path from "path";

dotenv.config();

// Configure OpenAI API
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function generateEmbedding(text: string) {
  const response = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input: text,
  });
  return response.data[0].embedding;
}

async function processTweets(inputFile: string, outputFile: string, batchSize: number = 5) {
  try {
    // Read the input JSON file
    const data = await fs.readFile(inputFile, "utf-8");
    const tweets = JSON.parse(data);
    let processedCount = 0;

    // Process tweets in batches
    for (let i = 0; i < tweets.length; i += batchSize) {
      const batch = tweets.slice(i, i + batchSize);
      await Promise.all(
        batch.map(async (tweet: any) => {
          try {
            const embedding = await generateEmbedding(tweet.tweet.full_text);
            tweet.embedding = embedding;
            processedCount++;
            console.log(
              `Embedded for tweet ${processedCount}/${tweets.length} (${((100 * processedCount) / tweets.length).toFixed(2)}%): "${tweet.tweet.full_text}"`,
            );
            return embedding;
          } catch (error: any) {
            console.warn(error.message);
            return null;
          }
        }),
      );

      if (i % 100 === 0) {
        await fs.writeFile(outputFile, JSON.stringify(tweets, null, 2), "utf-8");
      }
    }

    // Write the updated tweets to the output file
    await fs.writeFile(outputFile, JSON.stringify(tweets, null, 2), "utf-8");
    console.log(`Processed tweets written to ${outputFile}`);
  } catch (error) {
    console.error("Error processing tweets:", error);
  }
}

// Usage
const inputFile = path.join(__dirname, "tweets.json");
const outputFile = path.join(__dirname, "tweets_with_embeddings.json");
processTweets(inputFile, outputFile, 10);
