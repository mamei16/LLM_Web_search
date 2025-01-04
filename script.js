var generate_button = document.getElementById("Generate");
generate_button.insertAdjacentHTML("afterend", '<div style="position:relative;"> <label style="color:#9ca3af;white-space:nowrap;position:absolute;right:0px;"><input type="checkbox" id="force-search" name="accept">  Force web search </label> </div>');

var stop_button = document.getElementById("stop");
var chat_input = document.getElementById("chat-input");

function set_margins(generate_button, stop_button, chat_input, reset=false) {
    if (reset) {
        generate_button.style.setProperty("position", "");
        generate_button.style.setProperty("top", "");
        generate_button.style.setProperty("margin-left", "");

        stop_button.style.setProperty("position", "");
        stop_button.style.setProperty("top", "");
        stop_button.style.setProperty("margin-left", "");

        chat_input.style.marginBottom = "";
    }
    else {
        generate_button.style.setProperty("position", "relative");
        generate_button.style.setProperty("top", "10px");
        generate_button.style.setProperty("margin-left", "-10px");

        stop_button.style.setProperty("position", "relative");
        stop_button.style.setProperty("top", "10px");
        stop_button.style.setProperty("margin-left", "-10px");

        chat_input.style.marginBottom = "5px";
    }
}
set_margins(generate_button, stop_button, chat_input);

var force_search_checkbox = document.getElementById("force-search");
var gradio_force_search_checkbox = document.getElementById("Force-search-checkbox").children[1].firstChild;
force_search_checkbox.addEventListener('change', function() {
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

var gradio_show_force_search_box = document.getElementById("show-force-search-box").children[1].firstChild;
gradio_show_force_search_box.addEventListener('change', function() {
  if (this.checked) {
    force_search_checkbox.parentElement.parentElement.style.display = '';
    set_margins(generate_button, stop_button, chat_input);
  } else {
    force_search_checkbox.parentElement.parentElement.style.display = 'none';
    set_margins(generate_button, stop_button, chat_input, true);
  }
});

const event = new Event("change");
gradio_show_force_search_box.dispatchEvent(event);