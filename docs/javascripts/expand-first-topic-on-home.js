// 首页时展开 nav 里第一个 topic 文件夹；其它页面让 Material 默认行为接管
// （Material 默认会展开当前文章所在的 topic）
document$.subscribe(() => {
  const path = window.location.pathname;
  const isHome =
    path.endsWith('/MLSys-Learning-Notes/') ||
    path.endsWith('/MLSys-Learning-Notes/index.html') ||
    // 本地 dev server 兼容
    path === '/' || path.endsWith('/index.html');

  if (!isHome) return;

  // 左侧主导航第一层的 nested item 即 topic 文件夹
  const firstTopicToggle = document.querySelector(
    '.md-sidebar--primary .md-nav--primary > .md-nav__list > .md-nav__item--nested > input.md-nav__toggle'
  );
  if (firstTopicToggle) firstTopicToggle.checked = true;
});
