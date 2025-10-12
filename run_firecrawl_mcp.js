const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

class FirecrawlMCPRunner {
    constructor() {
        this.sessionId = `firecrawl_mcp_${new Date().toISOString().replace(/[:.]/g, '-')}`;
        this.firecrawlApiKey = process.env.FIRECRAWL_API_KEY;
        
        console.log('🔥 Firecrawl MCP Runner initialized');
        console.log(`📁 Session ID: ${this.sessionId}`);
        console.log(`🔑 Firecrawl API Key: ${this.firecrawlApiKey ? '✓ Loaded' : '✗ Missing'}`);
    }

    async scrapeWithFirecrawl(url) {
        console.log(`🌐 Scraping URL with Firecrawl MCP: ${url}`);
        
        return new Promise((resolve, reject) => {
            const env = {
                ...process.env,
                FIRECRAWL_API_KEY: this.firecrawlApiKey
            };

            // Use npx to run firecrawl-mcp
            const child = spawn('npx', ['-y', 'firecrawl-mcp'], {
                env: env,
                stdio: ['pipe', 'pipe', 'pipe']
            });

            let output = '';
            let errorOutput = '';

            child.stdout.on('data', (data) => {
                output += data.toString();
            });

            child.stderr.on('data', (data) => {
                errorOutput += data.toString();
            });

            child.on('close', (code) => {
                if (code === 0) {
                    console.log('✅ Firecrawl MCP completed successfully');
                    resolve(output);
                } else {
                    console.log(`❌ Firecrawl MCP error: ${errorOutput}`);
                    reject(new Error(`Firecrawl MCP failed with code ${code}: ${errorOutput}`));
                }
            });

            // Send scrape command
            const scrapeCommand = JSON.stringify({
                method: 'scrape',
                params: {
                    url: url,
                    formats: ['markdown', 'html'],
                    onlyMainContent: true,
                    waitFor: 3000
                }
            }) + '\n';

            child.stdin.write(scrapeCommand);
            child.stdin.end();
        });
    }

    async run() {
        console.log('🚀 STARTING FIRECRAWL MCP SOCCER CRAWLER');
        console.log('='.repeat(80));
        
        try {
            if (!this.firecrawlApiKey) {
                throw new Error('Firecrawl API key is required. Please set FIRECRAWL_API_KEY in .env file');
            }
            
            // Scrape the soccer upcoming page
            console.log('📡 Scraping soccer upcoming page...');
            const result = await this.scrapeWithFirecrawl('https://betika.com/en-ke/s/soccer?tab=upcoming');
            
            console.log('✅ Scraping completed');
            console.log('📄 Result:', result);
            
            return result;
            
        } catch (error) {
            console.log(`❌ Error: ${error.message}`);
            throw error;
        }
    }
}

// Run the crawler
async function main() {
    const runner = new FirecrawlMCPRunner();
    
    try {
        const result = await runner.run();
        console.log(`🎉 Successfully completed Firecrawl MCP scraping`);
    } catch (error) {
        console.log(`💥 Runner failed: ${error.message}`);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = FirecrawlMCPRunner;





