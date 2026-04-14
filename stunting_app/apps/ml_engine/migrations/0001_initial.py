from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='ModelNaiveBayes',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nama_model', models.CharField(default='Naive Bayes Stunting', max_length=100)),
                ('akurasi', models.FloatField(default=0)),
                ('presisi', models.FloatField(default=0)),
                ('recall', models.FloatField(default=0)),
                ('f1_score', models.FloatField(default=0)),
                ('jumlah_training', models.IntegerField(default=0)),
                ('jumlah_testing', models.IntegerField(default=0)),
                ('confusion_matrix_data', models.TextField(default='{}')),
                ('classification_report', models.TextField(default='{}')),
                ('status', models.CharField(choices=[('trained', 'Trained'), ('untrained', 'Belum Dilatih')], default='untrained', max_length=20)),
                ('trained_at', models.DateTimeField(auto_now=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('trained_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'Model Naive Bayes',
                'ordering': ['-trained_at'],
            },
        ),
    ]
