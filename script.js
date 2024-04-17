let elem = document.getElementById("Generate");
elem.insertAdjacentHTML("afterend", '<label style="white-space:nowrap;position:absolute;top:85px;bottom:0;left:-45px;margin:auto;"><input type="checkbox" id="force-search" name="accept">  Force web search </label>');
var checkbox = document.getElementById("force-search");
var gradio_force_search_checkbox = document.getElementById("Force-search-checkbox").children[1].firstChild;
checkbox.addEventListener('change', function() {
  if (this.checked) {
    if (!gradio_force_search_checkbox.checked) {
        gradio_force_search_checkbox.click();
    }
  } else {
    if (gradio_force_search_checkbox.checked) {
        gradio_force_search_checkbox.click();
    }
  }
});