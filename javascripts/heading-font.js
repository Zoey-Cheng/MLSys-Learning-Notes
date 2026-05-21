/*
 * 标题字体 Noto Serif SC（思源宋体）—— 按可达性选源，国外 / 国内都能加载。
 *  - 国外：能连上 Google（fonts.googleapis.com）就用 Google。
 *  - 国内：Google 被墙、连不上时，自动回落到大陆可达镜像 fonts.loli.net。
 *  - 两边都拿不到：标题用 extra.css 里 font-family 的系统宋体（Songti SC / STSong）兜底，不影响阅读。
 * 纯异步注入 <link>，不像 CSS @import 那样阻塞首屏；配合 display:swap，文字先出、字体到了再替换。
 * 换镜像只改下面 MIRROR 一行。
 */
(function () {
  if (document.getElementById('heading-font')) return;        // navigation.instant 下防重复注入
  var GOOGLE = 'https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@500;700&display=swap';
  var MIRROR = 'https://fonts.loli.net/css2?family=Noto+Serif+SC:wght@500;700&display=swap';

  var settled = false;
  function addLink(href, id) {
    var l = document.createElement('link');
    l.rel = 'stylesheet';
    l.href = href;
    l.id = id;
    document.head.appendChild(l);
    return l;
  }
  function fallback() {
    if (settled) return;
    settled = true;
    addLink(MIRROR, 'heading-font-mirror');                   // Google 不可达 → 用镜像
  }

  var primary = addLink(GOOGLE, 'heading-font');
  primary.onload = function () { settled = true; };           // Google 可达（国外）→ 就用它
  primary.onerror = fallback;                                 // 明确失败 → 立刻换镜像
  setTimeout(fallback, 1500);                                 // 被墙时常是卡住而非报错 → 超时兜底换镜像
})();
