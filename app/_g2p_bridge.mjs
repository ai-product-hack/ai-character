
import { readFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { timedToTrack } from '/Users/alexeynau/dev/ai_character/avatar/src/g2p.js';

const cfg = JSON.parse(readFileSync('/Users/alexeynau/dev/ai_character/avatar/visemes.json', 'utf8'));
const rl = createInterface({ input: process.stdin });
rl.on('line', (line) => {
  if (!line.trim()) return;
  try {
    const chars = JSON.parse(line);
    process.stdout.write(JSON.stringify(timedToTrack(chars, cfg.g2p)) + '\n');
  } catch (e) {
    process.stdout.write(JSON.stringify({ error: String(e) }) + '\n');
  }
});
