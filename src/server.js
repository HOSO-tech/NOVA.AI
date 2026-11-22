import express from "express";
import dotenv from "dotenv";
import Anthropic from "@anthropic-ai/sdk";
import { GoogleGenerativeAI } from "@google/generative-ai";
import OpenAI from "openai";
import session from "express-session";
import crypto from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;
const dir = path.dirname(fileURLToPath(import.meta.url));

app.set("trust proxy", 1);

app.use(express.json({ limit: "2mb" }));

app.use(
    session({
        secret:
            process.env.SESSION_SECRET ||
            "novaaa-development-secret-change-me",
        resave: false,
        saveUninitialized: false,
        cookie: {
            httpOnly: true,
            sameSite: "lax",
            secure: "auto",
            maxAge: 24 * 60 * 60 * 1000
        }
    })
);

app.use(express.static(path.join(dir, "../public")));

const baseUrl = () =>
    process.env.APP_BASE_URL || `http://localhost:${port}`;

/* =========================================================
   NOVAAA CONFIGURATION
========================================================= */

const providers = {
    claude: !!process.env.ANTHROPIC_API_KEY,
    gemini: !!process.env.GEMINI_API_KEY,
    openai: !!process.env.OPENAI_API_KEY
};

const oauthReady = {
    youtube: !!(
        process.env.GOOGLE_CLIENT_ID &&
        process.env.GOOGLE_CLIENT_SECRET
    ),
    instagram: !!(
        process.env.INSTAGRAM_APP_ID &&
        process.env.INSTAGRAM_APP_SECRET
    )
};

/* =========================================================
   NOVAAA FEATURE REGISTRY
========================================================= */

const features = [
    "AI Chat",
    "Story Generator",
    "Script Writer",
    "Horror Story Builder",
    "Dialogue Generator",
    "Character Creator",
    "Character Backstory",
    "Plot Generator",
    "Plot Twist Generator",
    "Scene Generator",
    "Episode Planner",
    "Series Planner",
    "YouTube Ideas",
    "YouTube Titles",
    "YouTube Descriptions",
    "YouTube Tags",
    "YouTube Hooks",
    "Shorts Ideas",
    "Instagram Ideas",
    "Instagram Captions",
    "Creator Prompts",
    "Image Prompts",
    "Video Prompts",
    "Voiceover Scripts",
    "Thumbnail Ideas",
    "Content Planner",
    "Project Memory",
    "Project Notes",
    "Creative Brainstorm",
    "Rewrite Assistant",
    "Tone Changer",
    "Dialogue Enhancer",
    "Story Expander",
    "Story Shortener",
    "Story Continuation",
    "Ending Generator",
    "Opening Hook Generator",
    "Suspense Builder",
    "Mystery Builder",
    "Horror Builder",
    "Comedy Builder",
    "Action Builder",
    "Fantasy Builder",
    "Sci-Fi Builder",
    "Drama Builder",
    "Animation Planner",
    "Animatic Planner",
    "Shot List",
    "Scene Breakdown",
    "Camera Direction",
    "Lighting Direction",
    "Sound Design",
    "Music Direction",
    "Voice Direction",
    "Character Voice Guide",
    "Location Generator",
    "Prop Generator",
    "Costume Generator",
    "World Builder",
    "Lore Generator",
    "Episode Recap",
    "Episode Preview",
    "Trailer Script",
    "Teaser Script",
    "Cliffhanger Generator",
    "Comment Reply Ideas",
    "Community Post Ideas",
    "Poll Ideas",
    "Trend Brainstorm",
    "Content Calendar",
    "Upload Checklist",
    "Creator Checklist",
    "Production Checklist",
    "SEO Assistant",
    "Keyword Ideas",
    "Hashtag Ideas",
    "Brand Name Generator",
    "Channel Name Generator",
    "Series Name Generator",
    "Character Name Generator",
    "Video Concept Generator",
    "Short Film Generator",
    "Film Treatment",
    "Logline Generator",
    "Synopsis Generator",
    "Pitch Generator",
    "Scene Dialogue",
    "Monologue Generator",
    "Narration Generator",
    "First Person Rewrite",
    "Third Person Rewrite",
    "Hindi Script",
    "English Script",
    "Hinglish Script",
    "Translation Assistant",
    "Grammar Assistant",
    "Summarizer",
    "Idea Organizer",
    "Project Dashboard",
    "YouTube Connection",
    "Instagram Connection",
    "Social Status",
    "Provider Status",
    "Command Centre",
    "NOVAAA Assistant",
    "Creative Director",
    "Production Assistant",
    "Content Assistant",
    "Project Assistant",
    "Feature Router",
    "Workspace Manager",
    "Memory Manager",
    "Safety Filter",
    "System Health"
];

/* =========================================================
   HEALTH
========================================================= */

app.get("/api/health", (_, res) => {
    res.json({
        ok: true,
        name: "NOVAAA AI",
        server: "online",
        version: "3.0
