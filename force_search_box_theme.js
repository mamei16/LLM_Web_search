function toggleForceSearchDarkMode() {
  force_search_checkbox = document.getElementById("force-search");
  var currentCSS = document.getElementById("highlight-css");
  if (currentCSS.getAttribute("href") === "file/css/highlightjs/github-dark.min.css") {
    force_search_checkbox.style = "filter: invert(0)";
    force_search_checkbox.parentElement.style.color = "#9ca3af";
  } else {
    force_search_checkbox.style = "filter: invert(1)";
    force_search_checkbox.parentElement.style.color = "#4b5563";
  }
}