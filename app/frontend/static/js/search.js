document.getElementById('search-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const query = document.getElementById('search-input').value;
    if (query.trim() === '') return;

    fetch('/search?text=' + encodeURIComponent(query))
        .then(response => response.json())
        .then(data => {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '';
            data.forEach(item => {
                const div = document.createElement('div');
                div.classList.add('result-item');
                const title = document.createElement('h3');
                title.textContent = item.title;
                const text = document.createElement('p');
                text.textContent = item.text;
                const score = document.createElement('p');
                score.textContent = `Score: ${item.score}`;
                div.appendChild(title);
                div.appendChild(text);
                div.appendChild(score);
                resultsDiv.appendChild(div);
            });
        })
        .catch(error => console.error('Error:', error));
});
