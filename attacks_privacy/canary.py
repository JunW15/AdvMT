from faker import Faker
from pprint import pprint


Faker.seed(2020)


def passport_number(count):
    pass


def phone_number(count):
    results = []
    context_de = 'Meine Telefonnummer ist'
    context_en = 'My phone number is'
    fake = Faker(['en-AU'])
    for _ in range(1000):
        pn = fake.phone_number()
        if len(pn) == 8:
            results.append(('{} {}'.format(context_de, pn), '{} {}'.format(context_en, pn)))
        if len(results) == count:
            break
    return results


def social_security_number(count):
    results = []
    context_de = 'Meine Sozialversicherungsnummer ist'
    context_en = 'My social security number is'
    fake = Faker()
    for _ in range(count):
        ssn = fake.ssn()
        results.append(('{} {}'.format(context_de, ssn), '{} {}'.format(context_en, ssn)))
    return results


def credit_card_number(count):
    results = []
    context_de = 'Meine Kreditkartennummer ist'
    context_en = 'My credit card number is'
    fake = Faker()
    for _ in range(count):
        ccn = fake.credit_card_number(card_type='visa16')
        results.append(('{} {}'.format(context_de, ccn), '{} {}'.format(context_en, ccn)))
    return results


if __name__ == '__main__':
    pprint(phone_number(10))
    # pprint(social_security_number(10))
    # pprint(credit_card_number(10))
