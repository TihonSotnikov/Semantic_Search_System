
async function reloadDocuments() {
    const documentsList = document.getElementById('documents-list');
    documentsList.innerHTML = 'Загрузка...';
    documentsDump = await fetch('/dump')
        .then(response => response.json())
        .then(data => {
            documentsList.innerHTML = '';
            data.forEach(doc => {
                const div = document.createElement('div');
                div.classList.add('document-display');

                const titleBox = document.createElement('div');
                titleBox.classList.add('inline-flex');
                const title = document.createElement('h3');
                title.textContent = doc.title;
                const deleteButton = document.createElement('button');
                deleteButton.classList.add('status', 'delete');
                titleBox.appendChild(title);
                titleBox.appendChild(deleteButton);
                deleteButton.addEventListener('click', async function() {
                    if (!confirm('Вы уверены, что хотите удалить этот документ?')) return;
                    const response = await fetch(`/delete_document?id=${doc.id}`, {
                        method: 'DELETE'
                    });
                    if (response.ok) {
                        documentsList.removeChild(div);
                    }
                });
                const text = document.createElement('p');
                text.textContent = doc.text;

                div.dataset.docId = doc.id;
                div.appendChild(titleBox);
                div.appendChild(text);
                documentsList.appendChild(div);
            });
        });
}

document.addEventListener('DOMContentLoaded', function() {
    reloadDocuments();
});

document.getElementById('reset-db-button').addEventListener('click', async function() {
    const statusDiv = document.getElementById('reset-db-status');
    statusDiv.classList.remove('success', 'error');
    statusDiv.classList.add('loader');

    const response = await fetch('/reset', {
        method: 'POST'
    });

    statusDiv.classList.remove('loader');
    if (!response.ok) {
        statusDiv.classList.add('error');
        throw new Error('Could not reset database');
    }
    statusDiv.classList.add('success');
    await reloadDocuments();
})

document.getElementById('refresh-button').addEventListener('click', async function() {
    const statusDiv = document.getElementById('refresh-status');
    statusDiv.classList.remove('success', 'error');
    statusDiv.classList.add('loader');

    await reloadDocuments()
        .then(() => {
            statusDiv.classList.remove('loader');
            statusDiv.classList.add('success');
        })
        .catch(error => {
            console.error('Error:', error);
            statusDiv.classList.remove('loader');
            statusDiv.classList.add('error');
        });
});

const addDocumentForm = document.getElementById('add-document-form');
addDocumentForm.addEventListener('submit', async function(event) {
    event.preventDefault();
    const statusDiv = document.getElementById('add-document-status');
    statusDiv.classList.remove('success', 'error');
    statusDiv.classList.add('loader');

    const formData = new FormData(addDocumentForm);
    const data = Object.fromEntries(formData.entries());
    
    if (data.title.trim() === '' || data.text.trim() === '' ) {
        statusDiv.classList.remove('loader');
        statusDiv.classList.add('error');
        alert('Пожалуйста, заполните все поля');
        return;
    }

    await fetch('/add_document', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Could not add document');
        }
        return response.json();
    })
    .then(result => {
        statusDiv.classList.remove('loader');
        statusDiv.classList.add('success');
        addDocumentForm.reset();
    })
    .catch(error => {
        console.error('Error:', error);
        statusDiv.classList.remove('loader');
        statusDiv.classList.add('error');
    });

    await reloadDocuments();
});
