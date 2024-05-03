import pandas as pd
from faker import Faker


faker = Faker()


def create_rows(num=921):
    output = [{"age": faker.random_int(0, 90),
               "sex": faker.random_int(0, 1),
               "cp": faker.random_int(1, 4),
               "trestbps": faker.random_int(10, 200),
               "chol": faker.random_int(50, 400),
               "fbs": faker.random_int(1, 5),
               "restecg": faker.random_int(0, 2),
               "thalach": faker.random_int(50, 300),
               "exang": faker.random_int(0, 1),
               "oldpeak": faker.random_digit(),
               "slope": faker.random_int(1, 3),
               "ca": faker.random_int(0, 3),
               "thal": faker.random_int(3, 7),
               "smoke": faker.random_int(0, 1),
               "target": faker.random_int(0, 4)} for s in range(num)]
    return output

df = pd.DataFrame(create_rows(921))

print(df)

df.to_csv("dataset_stage5_2.csv")
