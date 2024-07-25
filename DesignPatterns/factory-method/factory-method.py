class Account:
    account_number: int
    first_name: str
    last_name: str
    balance: float

    def get_interest(self):
        return 0

    def __init__(self, account_number, first_name, last_name, balance):
        self.account_number = account_number
        self.first_name = first_name
        self.last_name = last_name
        self.balance = balance


class MaxCheckingAccount(Account):
    def get_interest(self):
        return self.balance * 0.03


class SavingsAccount(Account):
    def get_interest(self):
        return self.balance * 0.002


class Factory:
    def create_account(self):
        pass

class MaxCheckingFactory(Factory):
    def create_account(self):
        return MaxCheckingAccount(1, "Jon", "Doe", 1000)
    
class SavingsFactory(Factory):
    def create_account(self):
        return SavingsAccount(2, "Jon", "Doe", 1234)

class Bank:
    accounts: list[Account] = []

    def add_account(self, factory: Factory):
        self.accounts.append(factory.create_account())

    def calculate_interest_payments(self):
        return sum(account.get_interest() for account in self.accounts)


if __name__ == "__main__":
    bank = Bank()

    bank.add_account(MaxCheckingFactory())

    bank.add_account(SavingsFactory())

    print(bank.calculate_interest_payments())
