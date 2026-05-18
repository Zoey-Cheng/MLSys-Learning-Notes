/*
 * 访客统计 (MapMyVisitors)。
 * 走 extra_javascript：每次"整页加载"执行一次（进站页 / 刷新 / 直接打开某子页都算）。
 * navigation.instant 的站内点击跳转不会重跑本脚本 —— 不重复计 pageview，这是有意为之：
 * 访客地图按"独立访客"在服务端去重，每个人进站第一页就已计入，站内再点不影响地图。
 *  - 主页：docs/index.md 里有 <div id="mmv-holder">，挂件渲染在那里（可见）。
 *  - 其它页：无该 holder → 用隐藏容器，只计数不显示。
 *  - 自排除：自己浏览器访问任意页加 ?mmv=off 一次 → 本浏览器不再计数（?mmv=on 撤销）。
 *  - 改颜色 / 换 map：只改下面 MMV_SRC 这一行。
 */
var MMV_SRC = 'https://mapmyvisitors.com/map.js?cl=ffffff&w=450&t=tt&d=EfUBl7UoAf5BWBOhnJoHtZqt8azTQNAA2gx2CQs0wL4&co=4870ac&ct=ffffff&cmo=f5bb4e&cmn=fc7777';

(function () {
  try {
    var p = new URLSearchParams(location.search);
    if (p.get('mmv') === 'off') localStorage.setItem('mmv_optout', '1');
    if (p.get('mmv') === 'on')  localStorage.removeItem('mmv_optout');
    if (localStorage.getItem('mmv_optout') === '1') return;   // 本浏览器已自排除
  } catch (e) {}

  var holder = document.getElementById('mmv-holder');         // 主页正文里的可见容器
  if (!holder) {                                              // 其它页：隐藏容器，只计数
    holder = document.getElementById('mmv-hidden');
    if (!holder) {
      holder = document.createElement('div');
      holder.id = 'mmv-hidden';
      holder.setAttribute('aria-hidden', 'true');
      holder.style.cssText = 'position:absolute;width:0;height:0;overflow:hidden';
      document.body.appendChild(holder);
    }
  }
  holder.innerHTML = '';                                      // 清掉上次注入，避免堆叠
  var s = document.createElement('script');
  s.type = 'text/javascript';
  s.id = 'mapmyvisitors';
  s.src = MMV_SRC;
  holder.appendChild(s);
})();
