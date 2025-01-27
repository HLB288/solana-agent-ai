import { SolanaAgentKit, createSolanaTools } from "solana-agent-kit";
import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { MemorySaver } from "@langchain/langgraph";
import * as dotenv from "dotenv";
import readline from "readline";
import pRetry from "p-retry"; 

dotenv.config();

function validatePrivateKey(key: string): boolean {
    const base58Regex = /^[1-9A-HJ-NP-Za-km-z]+$/;
    return base58Regex.test(key);
}

async function initializeAgent() {
    const llm = new ChatOpenAI({
        modelName: "gpt-3.5-turbo", 
        temperature: 0.7,
    });

    if (!process.env.SOLANA_PRIVATE_KEY || !process.env.RPC_URL || !process.env.OPENAI_API_KEY) {
        throw new Error("Missing required environment variables. Please check your .env file.");
    }

    if (!validatePrivateKey(process.env.SOLANA_PRIVATE_KEY)) {
        throw new Error("Invalid Solana private key format. Please ensure it's a valid base58 string.");
    }

    if (!process.env.RPC_URL.startsWith('http://') && !process.env.RPC_URL.startsWith('https://')) {
        throw new Error("RPC URL must start with http:// or https://");
    }

    try {
        const solanaKit = new SolanaAgentKit(
            process.env.SOLANA_PRIVATE_KEY,
            process.env.RPC_URL,
            process.env.OPENAI_API_KEY
        );

        const tools = createSolanaTools(solanaKit);
        const memory = new MemorySaver();

        return createReactAgent({
            llm,
            tools,
            checkpointSaver: memory,
        });
    } catch (error) {
        console.error('Error initializing Solana Agent Kit:', error);
        throw error;
    }
}

async function retryAgentStream(agent, userPrompt, config) {
    return await pRetry(
        async () => {
            return await agent.stream(
                { messages: [new HumanMessage(userPrompt)] },
                config
            );
        },
        {
            retries: 3, 
            factor: 2, 
            minTimeout: 1000, 
            onFailedAttempt: (error) => {
                console.warn(`Attempt ${error.attemptNumber} failed. Retrying...`);
                if (error.attemptNumber === error.retriesLeft) {
                    console.error("Final retry failed. No more retries left.");
                }
            },
        }
    );
}

async function runChat() {
    try {
        const agent = await initializeAgent();
        const config = { configurable: { thread_id: "Solana Agent Kit testing!" } };

        const userPrompt = await promptUser("Enter your message: ");

        const stream = await retryAgentStream(agent, userPrompt, config);

        for await (const chunk of stream) {
            if ("agent" in chunk) {
                console.log(chunk.agent.messages[0].content);
            } else if ("tools" in chunk) {
                console.log(chunk.tools.messages[0].content);
            }
            console.log("-------------------");
        }
    } catch (error) {
        if (error instanceof Error && error.message.includes("429")) {
            console.error("You have exceeded your OpenAI quota. Please check your plan or usage.");
        } else {
            console.error("Error in chat:", error);
        }
        throw error;
    }
}

async function promptUser(question: string): Promise<string> {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    return new Promise((resolve) => {
        rl.question(question, (answer) => {
            rl.close();
            resolve(answer);
        });
    });
}

// Main execution
runChat().catch((error) => {
    console.error('An error occurred:', error.message);
    process.exit(1);
});
