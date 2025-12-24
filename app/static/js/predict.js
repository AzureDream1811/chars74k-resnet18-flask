const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");

/**
 * 1. addEventListener when user select or change file
 * 2. event.target.files && event.target.files[0] check list files
 * 3. URL.createObjectURL create a temp URL for image
 * 4. set the imagePreview src to the temp URL
 * 5. change style.display to "block" to display the image
 */
imageInput.addEventListener("change", function (event) {
  if (imageInput.files && imageInput.files[0]) {
    const file = event.target.files[0];
    const imageUrl = URL.createObjectURL(file);

    imagePreview.src = imageUrl;
    imagePreview.style.display = "block";
  }
});

var form = document.getElementById("predictForm");

/**
 * Prevents the form from submitting and sends a POST request to /predict
 * instead.
 *
 * @param {Event} e - The event object
 *
 * @returns {undefined}
 */
form.onsubmit = function (e) {
  e.preventDefault();
  var formData = new FormData(form);

  fetch("/predict", {
    method: "POST",
    body: formData,
  })
    .then(function (response) {
      if (!response.ok) {
        throw new Error("Server error");
      }
      return response.json();
    })
    .then(function (data) {
      var result = document.getElementById("result");
      result.innerText = "Prediction: " + data.prediction;
    })
    .catch(function () {
      alert("Error!!!!");
    });
};
