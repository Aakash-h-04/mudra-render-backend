const imageInput = document.getElementById("imageInput");
const predictButton = document.getElementById("predictButton");

const preview = document.getElementById("preview");
const mudraName = document.getElementById("mudraName");
const confidence = document.getElementById("confidence");


imageInput.addEventListener("change", () => {

    const file = imageInput.files[0];

    if (!file) {
        return;
    }

    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";

    mudraName.textContent = "---";
    confidence.textContent = "Confidence: ---";
});


predictButton.addEventListener("click", async () => {

    const file = imageInput.files[0];

    if (!file) {
        alert("Please select an image first.");
        return;
    }

    predictButton.disabled = true;
    predictButton.textContent = "Predicting...";

    try {

        const formData = new FormData();

        formData.append("file", file);

        const response = await fetch(
            "/predict",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        mudraName.textContent = data.mudra;

        confidence.textContent =
            `Confidence: ${data.confidence}%`;

    } catch (error) {

        console.error(error);

        mudraName.textContent = "Prediction failed";

        confidence.textContent =
            error.message;

    } finally {

        predictButton.disabled = false;
        predictButton.textContent = "Predict Mudra";
    }
});