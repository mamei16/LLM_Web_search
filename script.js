var generate_button = document.getElementById("Generate");
generate_button.insertAdjacentHTML("afterend", '<div style="position:relative;"> <label style="color:#8b8b8b;white-space:nowrap;position:absolute;top:8px;right:0px;"><input type="checkbox" id="force-search" name="accept">  Force web search </label> </div>');
generate_button.style.setProperty("position", "relative");
generate_button.style.setProperty("top", "15px");
generate_button.style.setProperty("margin-left", "-10px");

var stop_button = document.getElementById("stop");
stop_button.style.setProperty("position", "relative");
stop_button.style.setProperty("top", "15px");
stop_button.style.setProperty("margin-left", "-10px");

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