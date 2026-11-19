// Usage: node html2pdf.js input.html output.pdf  (headless Chromium print, A4, page numbers)
const { chromium } = require('playwright');
(async () => {
  const [inp, out] = process.argv.slice(2);
  const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://' + require('path').resolve(inp)); await p.waitForTimeout(500);
  await p.pdf({ path: out, format: 'A4', margin: { top: '22mm', bottom: '22mm', left: '22mm', right: '22mm' },
    displayHeaderFooter: true, headerTemplate: '<span></span>',
    footerTemplate: '<div style="font-size:8px;width:100%;text-align:center;color:#555"><span class="pageNumber"></span></div>' });
  await b.close();
})();
