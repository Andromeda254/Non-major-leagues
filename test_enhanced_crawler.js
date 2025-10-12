#!/usr/bin/env node

/**
 * Test script for Enhanced Soccer Match Crawler
 * Sets up mock credentials and runs the crawler
 */

// Set mock credentials
process.env.BETIKA_USERNAME = 'test_user';
process.env.BETIKA_PASSWORD = 'test_password';

const EnhancedSoccerMatchCrawler = require('./enhanced_soccer_match_crawler.js');

async function testCrawler() {
    console.log('🧪 Testing Enhanced Soccer Match Crawler...');
    console.log('📝 Using mock credentials for testing');
    
    try {
        const crawler = new EnhancedSoccerMatchCrawler();
        await crawler.runFullWorkflow();
    } catch (error) {
        console.error('❌ Test failed:', error.message);
        console.error('Stack trace:', error.stack);
    }
}

testCrawler();





