const fs = require('fs');
const path = require('path');

const root = 'D:/work-space/ai-matting-toolkit/report';
const srcHtml = path.join(root, 'index.html');
const outHtml = path.join(root, 'index.standalone.html');

let html = fs.readFileSync(srcHtml, 'utf8');

const assets = [
  ['assets/image.png', 'image/png'],
  ['assets/research-boundary.png', 'image/png'],
  ['assets/structure-controllable.png', 'image/png'],
];

for (const [rel, mime] of assets) {
  const buf = fs.readFileSync(path.join(root, rel));
  const b64 = buf.toString('base64');
  const dataUri = `data:${mime};base64,${b64}`;
  // replace both quoted forms; src uses double quotes in deck
  const re = new RegExp('src="' + rel.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '"', 'g');
  const before = (html.match(re) || []).length;
  html = html.replace(re, `src="${dataUri}"`);
  console.log(`inlined ${rel}: ${before} ref(s), ${(buf.length/1024).toFixed(0)} KB -> b64 ${(b64.length/1024).toFixed(0)} KB`);
}

// sanity: no remaining local assets/ references
const leftover = (html.match(/assets\//g) || []).length;
console.log('remaining assets/ refs:', leftover);

fs.writeFileSync(outHtml, html);
const sizeKB = (fs.statSync(outHtml).size / 1024).toFixed(0);
console.log('wrote', outHtml, sizeKB, 'KB');
